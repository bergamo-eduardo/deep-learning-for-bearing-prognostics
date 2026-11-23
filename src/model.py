"""PyTorch dataset and fusion model for Paderborn bearing RUL."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


DatasetItem = Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float],
]
TENSOR_FORMAT_ERROR = "Tensores divergem das dimensões esperadas."


def _chunk_index(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Nome de chunk sem índice numérico: {path.name}")
    return int(match.group(1))


class BearingTensorDataset(Dataset):
    """Out-of-core dataset that loads one NPZ shard at a time."""

    def __init__(
        self,
        processed_dir: Path,
        bearings_list: List[str],
        metadata_path: Path,
        shuffle_shards: bool = False,
        return_meta: bool = False,
    ) -> None:
        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                metadata: Dict[str, object] = json.load(file)
            self.op_mean = np.asarray(metadata["op_mean"], dtype=np.float32).reshape(3)
            self.op_std = np.asarray(metadata["op_std"], dtype=np.float32).reshape(3)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise ValueError(TENSOR_FORMAT_ERROR)
        if not np.isfinite(self.op_mean).all() or not np.isfinite(self.op_std).all():
            raise ValueError(TENSOR_FORMAT_ERROR)
        if np.any(self.op_std <= 0.0):
            raise ValueError(TENSOR_FORMAT_ERROR)

        self.return_meta = return_meta
        all_files: List[Path] = []
        for bearing in bearings_list:
            bearing_dir = processed_dir / f"{bearing}_tensors"
            all_files.extend(
                sorted(
                    bearing_dir.glob("tensor_chunk_*.npz"),
                    key=_chunk_index,
                )
            )
        if shuffle_shards:
            random.shuffle(all_files)

        required_keys = {"X_vib", "X_op", "y_pct", "y_s", "file_idx"}
        self.samples: List[Tuple[Path, int]] = []
        for file_path in all_files:
            try:
                with np.load(file_path, allow_pickle=False) as data:
                    missing = required_keys.difference(data.files)
                    if missing:
                        raise ValueError(TENSOR_FORMAT_ERROR)
                    y_pct = data["y_pct"]
                    x_vib = data["X_vib"]
                    x_op = data["X_op"]
                    y_s = data["y_s"]
                    file_idx = data["file_idx"]
                    if y_pct.ndim != 1:
                        raise ValueError(TENSOR_FORMAT_ERROR)
                    if x_vib.ndim != 4 or x_vib.shape[1:] != (10, 2, 4096):
                        raise ValueError(TENSOR_FORMAT_ERROR)
                    if x_op.ndim != 2 or x_op.shape[1] != 3:
                        raise ValueError(TENSOR_FORMAT_ERROR)
                    if y_s.shape != y_pct.shape:
                        raise ValueError(TENSOR_FORMAT_ERROR)
                    if file_idx.shape != y_pct.shape or file_idx.dtype != np.int32:
                        raise ValueError(TENSOR_FORMAT_ERROR)
                    if x_vib.dtype != np.float32 or x_op.dtype != np.float32:
                        raise ValueError(TENSOR_FORMAT_ERROR)
                    if y_pct.dtype != np.float32 or y_s.dtype != np.float32:
                        raise ValueError(TENSOR_FORMAT_ERROR)
                    n_samples = int(y_pct.shape[0])
            except (KeyError, OSError, TypeError, ValueError):
                raise ValueError(TENSOR_FORMAT_ERROR)
            for internal_idx in range(n_samples):
                self.samples.append((file_path, internal_idx))

        self._cache_file: Path | None = None
        self._cache_data: Dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DatasetItem:
        file_path, internal_idx = self.samples[idx]
        if self._cache_file != file_path:
            with np.load(file_path, allow_pickle=False) as loaded:
                cached_data: Dict[str, np.ndarray] = {
                    "X_vib": loaded["X_vib"],
                    "X_op": loaded["X_op"],
                    "y_pct": loaded["y_pct"],
                    "y_s": loaded["y_s"],
                    "file_idx": loaded["file_idx"],
                }
            self._cache_file = file_path
            self._cache_data = cached_data

        if self._cache_data is None:
            raise RuntimeError("Falha ao inicializar o cache local do worker")

        cached_x_vib = self._cache_data["X_vib"]
        cached_x_op = self._cache_data["X_op"]
        cached_y_pct = self._cache_data["y_pct"]
        cached_y_s = self._cache_data["y_s"]
        cached_file_idx = self._cache_data["file_idx"]

        x_vib_raw = np.asarray(cached_x_vib[internal_idx], dtype=np.float32)
        x_vib_scaled = (
            x_vib_raw - np.mean(x_vib_raw, axis=-1, keepdims=True)
        ) / 10.0
        x_op_raw = np.asarray(cached_x_op[internal_idx], dtype=np.float32)
        x_op_scaled = (x_op_raw - self.op_mean) / self.op_std
        y_pct = torch.tensor(
            float(cached_y_pct[internal_idx]),
            dtype=torch.float32,
        )

        x_vib = torch.from_numpy(np.ascontiguousarray(x_vib_scaled))
        x_op = torch.from_numpy(np.ascontiguousarray(x_op_scaled))
        if not self.return_meta:
            return x_vib, x_op, y_pct

        return (
            x_vib,
            x_op,
            y_pct,
            int(cached_file_idx[internal_idx]),
            float(cached_y_s[internal_idx]),
        )


class FusionDegradationModel(nn.Module):
    """CNN-LSTM model for fusion of vibration and operating conditions."""

    def __init__(
        self,
        vib_channels: int = 2,
        op_features: int = 3,
        hidden_size: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if vib_channels <= 0:
            raise ValueError("vib_channels deve ser positivo")
        if op_features <= 0:
            raise ValueError("op_features deve ser positivo")
        if hidden_size <= 0:
            raise ValueError("hidden_size deve ser positivo")
        self.op_features = op_features
        self.hidden_size = hidden_size
        self.vib_channels = vib_channels
        self.cnn_block = nn.Sequential(
            nn.Conv1d(vib_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size + op_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x_vib: torch.Tensor,
        x_op: torch.Tensor,
    ) -> torch.Tensor:
        if x_vib.ndim != 4:
            raise ValueError(TENSOR_FORMAT_ERROR)
        batch_size, seq_len, channels, window_size = x_vib.shape
        if channels != self.vib_channels:
            raise ValueError(TENSOR_FORMAT_ERROR)
        cnn_input = x_vib.reshape(batch_size * seq_len, channels, window_size)
        cnn_out = self.cnn_block(cnn_input).squeeze(-1)
        lstm_in = cnn_out.reshape(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(lstm_in)
        last_time_step = lstm_out[:, -1, :]

        if x_op.ndim == 2:
            op_vec = x_op
        elif x_op.ndim == 3:
            op_vec = x_op[:, -1, :]
        else:
            raise ValueError(TENSOR_FORMAT_ERROR)
        if op_vec.shape[1] != self.op_features:
            raise ValueError(TENSOR_FORMAT_ERROR)

        fused = torch.cat((last_time_step, op_vec), dim=1)
        return self.regressor(fused).squeeze(-1)
