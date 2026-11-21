"""
Módulo de treinamento e validação.
Otimização de dados e uso de GPU.
"""

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importações dos módulos locais
from model import BearingTensorDataset, FusionDegradationModel
from preprocessing import calcular_estatisticas_treino


@dataclass(frozen=True)
class TrainConfig:
    """Hiperparâmetros centralizados do protocolo de treinamento."""

    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    num_workers: int = 2


def fusion_model_run(
    processed_dir: Path,
    bearings_train: List[str],
    bearings_val: List[str],
    metadata_path: Path,
    results_dir: Path,
    config: TrainConfig = TrainConfig(),
) -> float:
    """
    Treina o modelo com Early Stopping e MAE (L1Loss) como função de custo.
    """
    val_bearing = bearings_val[0] if bearings_val else "unknown"
    exp_results_dir = results_dir / f"val_{val_bearing}"
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    # Registro dos hiperparâmetros utilizados na iteração
    with open(exp_results_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=4)

    print("    [+] Carregando datasets...")
    train_dataset = BearingTensorDataset(processed_dir, bearings_train, metadata_path)
    val_dataset = BearingTensorDataset(processed_dir, bearings_val, metadata_path)

    # Cálculo estrito do particionamento populacional
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    n_total = n_train + n_val
    pct_train = (n_train / n_total) * 100
    pct_val = (n_val / n_total) * 100

    print(
        f"    [✔] Amostras: {n_train} (Treino: {pct_train:.2f}%) | "
        f"{n_val} (Validação: {pct_val:.2f}%)"
    )

    print("    [+] Configurando DataLoaders...")
    use_persistent = config.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    [+] Hardware: {device.type.upper()}")
    model = FusionDegradationModel().to(device)

    # Função de custo e otimizador
    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    print("            --- Treinamento ---\n")
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.epochs} [Train]",
            leave=False,
        )

        for X_vib, X_op, y in train_bar:
            X_vib = X_vib.to(device, non_blocking=True)
            X_op = X_op.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            predictions = model(X_vib, X_op)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_vib.size(0)
            train_bar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{config.epochs} [Val]",
            leave=False,
        )

        with torch.no_grad():
            for X_vib, X_op, y in val_bar:
                X_vib = X_vib.to(device, non_blocking=True)
                X_op = X_op.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                predictions = model(X_vib, X_op)
                loss = criterion(predictions, y)
                val_loss += loss.item() * X_vib.size(0)

                val_bar.set_postfix(loss=loss.item())

        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)

        print(
            f"            Época {epoch + 1:03d}/{config.epochs:03d} | "
            f"Train MAE: {train_loss:.6f} | Val MAE: {val_loss:.6f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_mae": train_loss,
            "val_mae": val_loss,
        })

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.patience:
            print(
                f"\n            [!] Early Stopping na época {epoch + 1}. "
                f"Sem melhoria por {config.patience} épocas."
            )
            break

    model.load_state_dict(best_model_wts)

    model_output_path = exp_results_dir / "best_model.pt"
    torch.save(model.state_dict(), model_output_path)

    history_output_path = exp_results_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_output_path, index=False)

    print("    [+] Gerando predições de validação...")
    model.eval()
    all_preds: List[float] = []
    all_targets: List[float] = []

    with torch.no_grad():
        for X_vib, X_op, y in val_loader:
            X_vib = X_vib.to(device, non_blocking=True)
            X_op = X_op.to(device, non_blocking=True)

            predictions = model(X_vib, X_op)

            if predictions.ndim == 0:
                all_preds.append(predictions.item())
                all_targets.append(y.item())
            else:
                all_preds.extend(predictions.cpu().numpy().tolist())
                all_targets.extend(y.numpy().tolist())

    y_max_global = train_dataset.y_max
    all_preds_np = np.array(all_preds, dtype=np.float32)
    all_targets_np = np.array(all_targets, dtype=np.float32)

    df_predictions = pd.DataFrame({
        "target_normalized": all_targets_np,
        "prediction_normalized": all_preds_np,
        "target_seconds": all_targets_np * y_max_global,
        "prediction_seconds": all_preds_np * y_max_global,
    })

    predictions_output_path = exp_results_dir / "validation_predictions.csv"
    df_predictions.to_csv(predictions_output_path, index=False)

    print(f"\n    [✔] Treinamento concluído. Resultados em: {exp_results_dir}")
    return best_val_loss


if __name__ == "__main__":
    defaults = TrainConfig()

    parser = argparse.ArgumentParser(description="Pipeline de Treinamento RUL CNN-LSTM")
    parser.add_argument(
        "--val",
        type=str,
        default="ALL",
        help="Ensaios para validação (ex: B09, B10) ou 'ALL' para LORO completa.",
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
    )

    projeto_root = Path.cwd()
    processed_dir = projeto_root / "data" / "processed_data"
    base_results_dir = projeto_root / "results"
    base_results_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    ensaios_all = [
        "B02", "B03", "B04", "B06", "B07", "B08", "B09",
        "B10", "B11", "B12", "B13", "B14", "B15", "B17",
    ]
    val_input = args.val.upper().strip()

    if val_input == "ALL":
        print("\n--- TREINAMENTO CROSS-VALIDATION (LORO) ---")
        ensaios_a_processar = ensaios_all
    else:
        val_targets = val_input.replace(",", " ").split()
        val_targets = sorted(set(v for v in val_targets if v))

        ensaios_invalidos = [v for v in val_targets if v not in ensaios_all]

        if ensaios_invalidos:
            print(f"\n[ERRO] Ensaios inválidos: {ensaios_invalidos}")
            print(f"Opções válidas: {ensaios_all}")
            sys.exit(1)

        print(f"\n--- TREINAMENTO PARCIAL: {val_targets} ---")
        ensaios_a_processar = val_targets

    print(
        f"[+] Configuração: epochs={config.epochs} | batch_size={config.batch_size} | "
        f"lr={config.lr} | weight_decay={config.weight_decay} | "
        f"patience={config.patience} | num_workers={config.num_workers}"
    )

    resultados_globais: Dict[str, float] = {}

    for val_bearing in ensaios_a_processar:
        lista_validacao = [val_bearing]
        lista_treino = [b for b in ensaios_all if b != val_bearing]

        exp_results_dir = base_results_dir / f"val_{val_bearing}"
        exp_results_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[+] Iteração de validação: {val_bearing}")
        print(f"    [+] Conjunto de treino: {lista_treino}")

        print("    [+] Calculando estatísticas de treino...")
        try:
            metadata_json_path = calcular_estatisticas_treino(
                processed_dir,
                exp_results_dir,
                lista_treino,
            )
        except Exception as e:
            print(f"    [ERRO] Falha nas estatísticas ({val_bearing}): {e}")
            continue

        print("    [+] Treinando modelo...")
        try:
            erro_best = fusion_model_run(
                processed_dir=processed_dir,
                bearings_train=lista_treino,
                bearings_val=lista_validacao,
                metadata_path=metadata_json_path,
                results_dir=base_results_dir,
                config=config,
            )

            resultados_globais[val_bearing] = erro_best
            print(f"    [✔] Fim da iteração {val_bearing}. Melhor MAE: {erro_best:.6f}")
            print("-" * 50)

        except RuntimeError as e:
            print(f"    [ERRO] Falha de execução em {val_bearing}: {e}")
            continue

    if len(resultados_globais) > 1:
        print("\n--- RESUMO DA VALIDAÇÃO ---")
        for ensaio, mae in resultados_globais.items():
            print(f" -> {ensaio}: MAE = {mae:.6f}")

        mae_medio = sum(resultados_globais.values()) / len(resultados_globais)
        print(f"\nMAE Global Médio: {mae_medio:.6f}\n")