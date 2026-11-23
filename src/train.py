"""Training program for the CNN-LSTM deep learning model."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import gc
import json
from pathlib import Path
import random
import shutil
import time
from typing import Dict, List

from model import BearingTensorDataset, FusionDegradationModel
import numpy as np
import pandas as pd
from preprocessing import calcular_estatisticas_treino
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

HISTORY_FIELDNAMES = [
    "epoch",
    "timestamp",
    "epoch_time_s",
    "lr",
    "train_mae_norm",
    "val_mae_norm",
    "train_mae_pct",
    "val_mae_pct",
    "best_val_mae_norm",
    "is_best",
    "epochs_sem_melhora",
]


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters shared by CLI and training functions."""

    bearings: List[str] = field(default_factory=lambda: [
        "B02", "B03", "B04", "B06", "B07", "B08", "B09", "B10",
        "B11", "B12", "B13", "B14", "B15", "B17",
    ])
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_factor: float = 0.5
    lr_patience: int = 4
    min_lr: float = 1e-5
    max_grad_norm: float = 1.0
    patience: int = 15
    num_workers: int = 4
    seed: int = 1


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_seeds(value: str) -> List[int]:
    seeds: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            seed = int(item)
            if seed not in seeds:
                seeds.append(seed)
    if not seeds:
        raise ValueError("A opção --seeds deve conter pelo menos uma semente")
    return seeds


def _parse_validation(value: str, available_bearings: List[str]) -> List[str]:
    if value.upper().strip() == "ALL":
        return available_bearings
    bearings: List[str] = []
    for item in value.upper().split(","):
        bearing = item.strip()
        if bearing and bearing not in bearings:
            bearings.append(bearing)
    invalid = [
        bearing for bearing in bearings if bearing not in available_bearings
    ]
    if invalid:
        raise ValueError(
            f"Rolamentos não configurados: {invalid}. "
            f"Opções válidas: {available_bearings}"
        )
    if not bearings:
        raise ValueError("A opção --val não contém rolamentos")
    return bearings


def _init_epoch_log(log_path: Path) -> None:
    """Create or overwrite the epoch CSV log with its header."""
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDNAMES)
        writer.writeheader()


def _append_epoch_log(log_path: Path, row: Dict[str, object]) -> None:
    """Append one epoch record and close immediately for safe live inspection."""
    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDNAMES)
        writer.writerow(row)
        file.flush()


def fusion_model_run(
    processed_dir: Path,
    bearings_train: List[str],
    bearings_val: List[str],
    metadata_path: Path,
    results_dir: Path,
    config: TrainConfig,
) -> tuple[float, pd.DataFrame]:
    """Train one seed/bearing fold and return the best validation metric."""
    set_seed(config.seed)
    val_bearing = bearings_val[0]
    exp_results_dir = results_dir / f"val_{val_bearing}"
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    config_payload = asdict(config)
    with (exp_results_dir / "train_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(config_payload, file, indent=4)
    metadata_output_path = exp_results_dir / "metadata_normalization.json"
    if metadata_path.resolve() != metadata_output_path.resolve():
        shutil.copy2(metadata_path, metadata_output_path)

    history_path = exp_results_dir / "training_history.csv"
    best_model_path = exp_results_dir / "best_model.pt"
    _init_epoch_log(history_path)

    train_dataset = BearingTensorDataset(
        processed_dir,
        bearings_train,
        metadata_path,
    )
    val_dataset = BearingTensorDataset(
        processed_dir,
        bearings_val,
        metadata_path,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Treino ou validação não contém sequências")

    persistent_workers = config.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    num_op_features = int(metadata["num_op_features"])
    model = FusionDegradationModel(
        vib_channels=2,
        op_features=num_op_features,
    ).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
        min_lr=config.min_lr,
    )

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_sem_melhora = 0

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()

        model.train()
        train_loss = 0.0
        train_samples = 0
        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.epochs} [Train]",
            leave=False,
        )
        for x_vib, x_op, y_pct in train_bar:
            x_vib = x_vib.to(device, non_blocking=True)
            x_op = x_op.to(device, non_blocking=True)
            y_pct = y_pct.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(x_vib, x_op)
            loss = criterion(predictions, y_pct)
            loss.backward()
            if config.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.max_grad_norm,
                )
            optimizer.step()
            batch_size = int(x_vib.size(0))
            train_loss += float(loss.item()) * batch_size
            train_samples += batch_size
            train_bar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        val_samples = 0
        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{config.epochs} [Val]",
            leave=False,
        )
        with torch.no_grad():
            for x_vib, x_op, y_pct in val_bar:
                x_vib = x_vib.to(device, non_blocking=True)
                x_op = x_op.to(device, non_blocking=True)
                y_pct = y_pct.to(device, non_blocking=True)
                predictions = model(x_vib, x_op)
                loss = criterion(predictions, y_pct)
                batch_size = int(x_vib.size(0))
                val_loss += float(loss.item()) * batch_size
                val_samples += batch_size
                val_bar.set_postfix(loss=loss.item())

        train_mae_norm = train_loss / train_samples
        val_mae_norm = val_loss / val_samples
        epoch_time_s = time.perf_counter() - epoch_start
        current_lr = float(optimizer.param_groups[0]["lr"])

        is_best = val_mae_norm < best_val_loss
        if is_best:
            best_val_loss = val_mae_norm
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, best_model_path)
            epochs_sem_melhora = 0
        else:
            epochs_sem_melhora += 1

        _append_epoch_log(
            history_path,
            {
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "epoch_time_s": round(epoch_time_s, 3),
                "lr": current_lr,
                "train_mae_norm": train_mae_norm,
                "val_mae_norm": val_mae_norm,
                "train_mae_pct": train_mae_norm * 100.0,
                "val_mae_pct": val_mae_norm * 100.0,
                "best_val_mae_norm": best_val_loss,
                "is_best": int(is_best),
                "epochs_sem_melhora": epochs_sem_melhora,
            },
        )

        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} | "
            f"LR: {current_lr:.2e} | "
            f"Train MAE: {train_mae_norm:.6f} | Val MAE: {val_mae_norm:.6f} | "
            f"Time: {epoch_time_s:.1f}s"
        )
        if epochs_sem_melhora >= config.patience:
            print(
                f"Early stopping na época {epoch}: "
                f"{config.patience} épocas sem melhoria."
            )
            break
        scheduler.step(val_mae_norm)
        new_lr = float(optimizer.param_groups[0]["lr"])
        if new_lr < current_lr:
            print(
                f"  -> ReduceLROnPlateau: LR reduzido de {current_lr:.2e} "
                f"para {new_lr:.2e}."
            )
        
    del train_bar, val_bar, train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.load_state_dict(best_model_wts)
    if not best_model_path.exists():
        torch.save(best_model_wts, best_model_path)

    val_meta_dataset = BearingTensorDataset(
        processed_dir,
        bearings_val,
        metadata_path,
        shuffle_shards=False,
        return_meta=True,
    )
    val_eval_loader = DataLoader(
        val_meta_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    all_file_idx: List[int] = []
    all_true_pct: List[float] = []
    all_pred_pct: List[float] = []
    all_true_s: List[float] = []
    model.eval()
    with torch.no_grad():
        for x_vib, x_op, y_pct, file_idx, y_s in val_eval_loader:
            x_vib = x_vib.to(device, non_blocking=True)
            x_op = x_op.to(device, non_blocking=True)
            y_pred_norm = model(x_vib, x_op).detach().cpu().numpy()
            y_true_pct = y_pct.numpy() * 100.0
            y_pred_pct = y_pred_norm * 100.0
            all_true_pct.extend(y_true_pct.tolist())
            all_pred_pct.extend(y_pred_pct.tolist())
            all_file_idx.extend(list(file_idx.numpy().tolist()))
            all_true_s.extend(y_s.numpy().tolist())

    df_predictions = pd.DataFrame({
        "file_idx": all_file_idx,
        "y_true_pct": all_true_pct,
        "y_pred_pct": all_pred_pct,
        "y_pred_pct_clipped": np.clip(all_pred_pct, 0.0, 100.0).tolist(),
        "y_true_s": all_true_s,
    })
    df_predictions.to_csv(
        exp_results_dir / "validation_predictions.csv",
        index=False,
    )

    df_cycle = df_predictions.groupby("file_idx", as_index=False).agg({
        "y_pred_pct": "mean",
        "y_pred_pct_clipped": "mean",
        "y_true_pct": "first",
        "y_true_s": "first",
    })
    df_cycle.to_csv(
        exp_results_dir / "validation_predictions_by_cycle.csv",
        index=False,
    )

    mae_seq = float(
        np.mean(
            np.abs(
                df_predictions["y_true_pct"].to_numpy() - df_predictions["y_pred_pct"].to_numpy()
            )
        )
    )
    mae_cycle = float(
        np.mean(
            np.abs(
                df_cycle["y_true_pct"].to_numpy() - df_cycle["y_pred_pct"].to_numpy()
            )
        )
    )
    mae_cycle_clipped = float(
        np.mean(
            np.abs(
                df_cycle["y_true_pct"].to_numpy() - df_cycle["y_pred_pct_clipped"].to_numpy()
            )
        )
    )
    metrics = pd.DataFrame([{
        "val_bearing": val_bearing,
        "seed": config.seed,
        "mae_pct_seq": mae_seq,
        "mae_pct_cycle": mae_cycle,
        "mae_pct_cycle_clipped": mae_cycle_clipped,
    }])
    return best_val_loss, metrics


def _run_one_seed(
    seed: int,
    val_bearings: List[str],
    processed_dir: Path,
    base_results_dir: Path,
    base_config: TrainConfig,
) -> pd.DataFrame:
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    config = replace(base_config, seed=seed)
    seed_dir = base_results_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    rows: List[pd.DataFrame] = []
    for val_bearing in val_bearings:
        train_bearings = [
            bearing for bearing in config.bearings if bearing != val_bearing
        ]
        val_dir = seed_dir / f"val_{val_bearing}"
        val_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = calcular_estatisticas_treino(
            processed_dir,
            val_dir,
            train_bearings,
        )
        _, metrics = fusion_model_run(
            processed_dir=processed_dir,
            bearings_train=train_bearings,
            bearings_val=[val_bearing],
            metadata_path=metadata_path,
            results_dir=seed_dir,
            config=config,
        )
        rows.append(metrics)

    summary = pd.concat(rows, ignore_index=True)
    mean_row = pd.DataFrame([{
        "val_bearing": "MEAN_GLOBAL",
        "seed": seed,
        "mae_pct_seq": summary["mae_pct_seq"].mean(),
        "mae_pct_cycle": summary["mae_pct_cycle"].mean(),
        "mae_pct_cycle_clipped": summary["mae_pct_cycle_clipped"].mean(),
    }])
    summary = pd.concat([summary, mean_row], ignore_index=True)
    summary.to_csv(seed_dir / f"loro_summary_seed_{seed}.csv", index=False)
    return summary


if __name__ == "__main__":
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Treinamento CNN-LSTM para RUL percentual"
    )
    parser.add_argument("--val", type=str, default="ALL")
    parser.add_argument("--seeds", type=str, default=str(defaults.seed))
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument(
        "--weight-decay", type=float, default=defaults.weight_decay
    )
    parser.add_argument("--lr-factor", type=float, default=defaults.lr_factor)
    parser.add_argument("--lr-patience", type=int, default=defaults.lr_patience)
    parser.add_argument("--min-lr", type=float, default=defaults.min_lr)
    parser.add_argument(
        "--max-grad-norm", type=float, default=defaults.max_grad_norm
    )
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    args = parser.parse_args()

    base_config = TrainConfig(
        bearings=list(defaults.bearings),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        min_lr=args.min_lr,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        num_workers=args.num_workers,
    )
    val_bearings = _parse_validation(args.val, base_config.bearings)
    seeds = _parse_seeds(args.seeds)

    projeto_root = Path(__file__).resolve().parents[1]
    processed_dir = projeto_root / "data" / "processed_data"
    base_results_dir = projeto_root / "results"
    run_results_dir = base_results_dir / (
        "bearings_" + "-".join(base_config.bearings)
    )
    run_results_dir.mkdir(parents=True, exist_ok=True)

    all_seed_summaries: List[pd.DataFrame] = []
    for seed in seeds:
        summary = _run_one_seed(
            seed=seed,
            val_bearings=val_bearings,
            processed_dir=processed_dir,
            base_results_dir=run_results_dir,
            base_config=base_config,
        )
        all_seed_summaries.append(summary)

    if len(seeds) > 1:
        combined = pd.concat(all_seed_summaries, ignore_index=True)
        multiseed = combined.groupby("val_bearing", as_index=False).agg(
            mae_pct_cycle_mean=("mae_pct_cycle", "mean"),
            mae_pct_cycle_std=("mae_pct_cycle", "std"),
        )
        multiseed.to_csv(
            run_results_dir / "loro_summary_multiseed.csv",
            index=False,
        )
