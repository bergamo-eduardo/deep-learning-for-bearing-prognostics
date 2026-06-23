"""
Módulo de Treinamento e Validação (Intra-Ensaio) - Versão de Depuração Rápida
Aviso Metodológico: O uso deste script implica em Data Leakage por sobreposição de 
janelas deslizantes. Uso restrito à depuração computacional do modelo.
"""

import copy
import json
from pathlib import Path
from typing import List, Dict
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# Importação explícita do Subset no escopo global para garantir compatibilidade
from torch.utils.data import DataLoader, Dataset, Subset

# Importações dos módulos locais
from model import BearingTensorDataset, FusionDegradationModel
from preprocessing import calcular_estatisticas_treino

def fusion_model_run(
    train_dataset: Subset,
    val_dataset: Subset,
    y_max_global: float,
    exp_results_dir: Path,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 10
) -> float:
    """
    Executa a otimização matemática do modelo recebendo subconjuntos (Subsets) pré-particionados.
    """
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    print("    [+] Configurando DataLoaders iteráveis...")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    [+] Alocando arquitetura no hardware selecionado ({device.type.upper()})...")
    model = FusionDegradationModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        for X_vib, X_op, y in train_bar:
            X_vib, X_op, y = X_vib.to(device, non_blocking=True), X_op.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            predictions = model(X_vib, X_op)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_vib.size(0)
            train_bar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for X_vib, X_op, y in val_bar:
                X_vib, X_op, y = X_vib.to(device, non_blocking=True), X_op.to(device, non_blocking=True), y.to(device, non_blocking=True)
                
                predictions = model(X_vib, X_op)
                loss = criterion(predictions, y)
                val_loss += loss.item() * X_vib.size(0)
                val_bar.set_postfix(loss=loss.item())

        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)

        print(f"            Época {epoch+1:03d}/{epochs:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        history.append({"epoch": epoch + 1, "train_mse": train_loss, "val_mse": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"\n            [!] Parada antecipada acionada. Sem melhoria por {patience} épocas.")
            break

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), exp_results_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(exp_results_dir / "training_history.csv", index=False)

    print("    [+] Gerando predições finais...")
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for X_vib, X_op, y in val_loader:
            X_vib, X_op = X_vib.to(device, non_blocking=True), X_op.to(device, non_blocking=True)
            predictions = model(X_vib, X_op)
            all_preds.extend(predictions.cpu().numpy().flatten().tolist())
            all_targets.extend(y.numpy().flatten().tolist())

    df_predictions = pd.DataFrame({
        "target_normalized": all_targets,
        "prediction_normalized": all_preds,
        "target_seconds": np.array(all_targets) * y_max_global,
        "prediction_seconds": np.array(all_preds) * y_max_global
    })
    
    # Ordenação monotônica temporal para assegurar a consistência visual da reta decrescente
    df_predictions = df_predictions.sort_values(by="target_seconds", ascending=False).reset_index(drop=True)
    df_predictions.to_csv(exp_results_dir / "validation_predictions.csv", index=False)

    return best_val_loss

if __name__ == "__main__":
    # =========================================================================
    # CONFIGURAÇÃO DE PARÂMETROS BRUTOS (HARDCODED)
    # =========================================================================
    ENSAIO_CONFIG: str = "B04"       # Identificador nominal do ensaio
    SPLIT_CONFIG: float = 0.75        # Fração aproximada destinada ao conjunto de treinamento
    TAMANHO_BLOCO: int = 10          # Comprimento do bloco sequencial contínuo
    # =========================================================================

    projeto_root = Path.cwd()
    processed_dir = projeto_root / "data" / "processed_data"
    base_results_dir = projeto_root / "results"
    
    ensaio_alvo = ENSAIO_CONFIG.upper().strip()
    exp_results_dir = base_results_dir / f"val_s{ensaio_alvo}"
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f" TREINAMENTO INTRA-ENSAIO ESTÁTICO: {ensaio_alvo}")
    print(f" Proporção Alvo de Particionamento: {SPLIT_CONFIG}")
    print(" ATENÇÃO: Risco elevado de Data Leakage (Uso apenas para debug)")
    print(f"=======================================================\n")

    # 1. Extração de estatísticas utilizando o próprio ensaio (introduz viés)
    try:
        metadata_json_path = calcular_estatisticas_treino(
            processed_dir, 
            exp_results_dir, 
            [ensaio_alvo]
        )
    except Exception as e:
        print(f"[ERRO] Falha ao extrair estatísticas: {e}")
        sys.exit(1)

    # 2. Instanciação do Dataset Completo
    full_dataset = BearingTensorDataset(processed_dir, [ensaio_alvo], metadata_json_path)
    y_max_global = full_dataset.y_max

    # =========================================================================
    # PARTICIONAMENTO CRONOLÓGICO POR BLOCOS (STRIDED BLOCK SPLIT)
    # =========================================================================
    total_size = len(full_dataset)
    indices_totais = list(range(total_size))
    
    indices_treino: List[int] = []
    indices_val: List[int] = []
    
    # Interpolação estruturada de blocos sequenciais contínuos
    # Garante a amostragem homogênea de toda a trajetória de degradação
    for i in range(0, total_size, TAMANHO_BLOCO * 2):
        indices_treino.extend(indices_totais[i : i + TAMANHO_BLOCO])
        indices_val.extend(indices_totais[i + TAMANHO_BLOCO : i + TAMANHO_BLOCO * 2])
        
    # Filtragem terminal para contenção de limites dimensionais
    indices_treino = [idx for idx in indices_treino if idx < total_size]
    indices_val = [idx for idx in indices_val if idx < total_size]
    
    # Instanciação dos objetos concretos do tipo Subset
    train_dataset = Subset(full_dataset, indices_treino)
    val_dataset = Subset(full_dataset, indices_val)

    print(f"    [✔] Dataset particionado por blocos temporais intercalados:")
    print(f"        -> Treino: {len(train_dataset)} amostras | Validação: {len(val_dataset)} amostras")
    # =========================================================================

    # 4. Execução do Treinamento
    try:
        erro_best = fusion_model_run(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            y_max_global=y_max_global,
            exp_results_dir=exp_results_dir,
            epochs=50,
            batch_size=64,
            lr=0.001,
            patience=10
        )
        print(f"\n[✔] OTIMIZAÇÃO CONCLUÍDA. MSE Mínimo Obtido: {erro_best:.6f}")
    except RuntimeError as e:
        print(f"\n[ERRO DE TEMPO DE EXECUÇÃO]:\n{e}")