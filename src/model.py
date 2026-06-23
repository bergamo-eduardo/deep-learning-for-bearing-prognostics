"""
Módulo de Arquitetura - Modelo Híbrido CNN-LSTM e Dataset Customizado
"""

import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import glob
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

class BearingTensorDataset(Dataset):
    """
    Dataset out-of-core otimizado para leitura de tensores temporais 4D em disco.
    Aplica normalização local (Z-score por janela) nos canais cinematizados de vibração
    e normalização global z-score/linear nas variáveis de processo e RUL.
    """
    def __init__(self, processed_dir: Path, bearings_list: List[str], metadata_path: Path, shuffle_shards: bool = False):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata: Dict = json.load(f)
            
        self.op_mean = np.array(metadata["op_mean"], dtype=np.float32)
        self.op_std = np.array(metadata["op_std"], dtype=np.float32)
        self.y_max = float(metadata["y_max"])

        self.samples: List[Tuple[str, int]] = []
        all_files: List[str] = []
        
        # Consolidação e ordenação determinística dos caminhos dos shards
        for exp in bearings_list:
            tensor_path = processed_dir / f"{exp}_tensors"
            files = sorted(glob.glob(str(tensor_path / "*.npz")))
            all_files.extend(files)

        # Permutação de shards para otimização de I/O linear sob amostragem não-aleatória global
        if shuffle_shards:
            random.shuffle(all_files)

        # Indexação lazy dos metadados estruturais via lazy-load estruturado
        for file in all_files:
            with np.load(file) as data:
                n_samples = data["X"].shape[0]
            for internal_idx in range(n_samples):
                self.samples.append((file, internal_idx))

        # Inicialização do mecanismo de cache em nível de processo/worker
        self.last_file: Optional[str] = None
        self.last_data: Optional[Dict[str, np.ndarray]] = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        file, internal_idx = self.samples[idx]
        
        # Gerenciamento de cache síncrono para mitigação de overhead de paginação de disco
        if self.last_file != file:
            self.last_data = dict(np.load(file))
            self.last_file = file
            
        # Type guarding para análise estática de tipo (Pylance/Pyright verification)
        if self.last_data is None:
            raise RuntimeError(f"Falha de consistência no runtime: Buffer do arquivo {file} não inicializado.")

        X = self.last_data["X"][internal_idx]
        y = self.last_data["y"][internal_idx]

        # Apenas remoção da componente contínua (DC offset) preservando a amplitude:
        X_vib_raw = X[:, 0:2, :]
        vib_mean = np.mean(X_vib_raw, axis=-1, keepdims=True)

        # Fator de conversão global empírico ou limite físico em gravidade (g)
        # Evite a divisão por desvio padrão local.
        X_vib_scaled = (X_vib_raw - vib_mean) / 10.0 # Exemplo de divisão por fator fixo de escala
        X_vib = torch.tensor(X_vib_scaled, dtype=torch.float32)

        # Fusão de Sensores: Padronização Z-score global das variáveis de processo
        X_op_raw = X[:, 2:4, 0]
        X_op_scaled = (X_op_raw - self.op_mean) / self.op_std
        X_op = torch.tensor(X_op_scaled, dtype=torch.float32)

        # Normalização linear contínua da variável dependente (RUL Target)
        y_tensor = torch.tensor(y / self.y_max, dtype=torch.float32)

        return X_vib, X_op, y_tensor


class FusionDegradationModel(nn.Module):
    """
    Arquitetura de fusão de sensores baseada em extratores convolucionais espaciais
    associados a uma unidade recorrente LSTM para modelagem de trajetórias temporais.
    """
    def __init__(self, vib_channels: int = 2, op_features: int = 2):
        super(FusionDegradationModel, self).__init__()

        self.cnn_block = nn.Sequential(
            nn.Conv1d(in_channels=vib_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        hidden_size = 64
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        fusion_size = hidden_size + op_features
        self.regressor = nn.Sequential(
            nn.Linear(fusion_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_vib: torch.Tensor, x_op: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, window_size = x_vib.size()
        
        x_vib = x_vib.view(batch_size * seq_len, channels, window_size)
        
        cnn_out = self.cnn_block(x_vib)
        cnn_out = cnn_out.squeeze(-1)
        
        lstm_in = cnn_out.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(lstm_in)
        last_time_step = lstm_out[:, -1, :]
        
        last_op_step = x_op[:, -1, :]
        fused_features = torch.cat((last_time_step, last_op_step), dim=1)
        
        rul_pred = self.regressor(fused_features)
        return rul_pred.squeeze()