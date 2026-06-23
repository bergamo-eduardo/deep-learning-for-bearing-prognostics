"""
Módulo de Treinamento Rápido e Validação de Setup
Otimização do fluxo de dados, tensores e uso de GPU.
"""

import copy
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Importações dos módulos locais
from model import BearingTensorDataset, FusionDegradationModel
from preprocessing import calcular_estatisticas_treino

def fusion_model_run(
    processed_dir: Path,
    bearings_train: List[str],
    bearings_val: List[str],
    metadata_path: Path,
    results_dir: Path,
    epochs: int = 50,         # Limite máximo de épocas elevado
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 10        # Critério de tolerância para interrupção precoce
) -> float:
    """
    Executa a otimização matemática do modelo com suporte a Early Stopping,
    registra o histórico de perdas por época e exporta os artefatos finais.
    """
    val_bearing = bearings_val[0] if bearings_val else "unknown"
    exp_results_dir = results_dir / f"val_{val_bearing}"
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    print("    [+] Inicializando instâncias do Dataset (Treino e Validação)...")
    train_dataset = BearingTensorDataset(processed_dir, bearings_train, metadata_path)
    val_dataset = BearingTensorDataset(processed_dir, bearings_val, metadata_path)
    print(f"    [✔] Datasets instanciados: {len(train_dataset)} amostras (Treino) | {len(val_dataset)} amostras (Validação)")

    print("    [+] Configurando DataLoaders iteráveis...")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=6, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=6, pin_memory=True, persistent_workers=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    [+] Alocando arquitetura no hardware selecionado ({device.type.upper()})...")
    model = FusionDegradationModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print("            --- Iniciando treinamento ---\n")
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0  # Contador para o Early Stopping
    
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
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
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        
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

        print(f"            Época {epoch+1:03d}/{epochs:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        history.append({
            "epoch": epoch + 1,
            "train_mse": train_loss,
            "val_mse": val_loss
        })

        # Lógica de Early Stopping e salvamento do melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"\n            [!] Parada antecipada acionada. Sem melhoria no Erro de Validação por {patience} épocas consecutivas.")
            break

    # Consolidação estrutural e gravação dos artefatos em disco
    model.load_state_dict(best_model_wts)
    
    model_output_path = exp_results_dir / "best_model.pt"
    torch.save(model.state_dict(), model_output_path)
    
    history_output_path = exp_results_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_output_path, index=False)

    print("    [+] Gerando predições finais para análise de fim de vida útil...")
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
        "prediction_seconds": all_preds_np * y_max_global
    })
    
    predictions_output_path = exp_results_dir / "validation_predictions.csv"
    df_predictions.to_csv(predictions_output_path, index=False)

    print(f"\n    [✔] Processo concluído. Artefatos exportados para: {exp_results_dir}")
    return best_val_loss


if __name__ == "__main__":
    import argparse
    import sys

    # 1. Configuração de Argumentos via Linha de Comando
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento RUL CNN-LSTM")
    # Removido o nargs="+" e o default em formato de lista para evitar confusão de tipagem.
    # O argumento agora aceita uma string única que pode conter separadores (ex: "B09" ou "B09,B10" ou "ALL")
    parser.add_argument(
        "--val", 
        type=str, 
        default="ALL", 
        help="Especifique os ensaios para validação (ex: B09 ou B09,B10) ou 'ALL' para LORO completa."
    )
    args = parser.parse_args()

    projeto_root = Path.cwd()
    processed_dir = projeto_root / "data" / "processed_data"
    base_results_dir = projeto_root / "results"
    base_results_dir.mkdir(parents=True, exist_ok=True)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 2. Definição da População de Ensaios Não-Censurados
    ensaios_all = ["B02", "B04", "B09", "B10", "B11", "B12", "B17"]

    # 3. Tratamento Robusto da Lógica de Execução
    val_input = args.val.upper().strip()

    if val_input == "ALL":
        print("\n=======================================================")
        print("   TREINAMENTO CROSS-VALIDATION LORO (Leave-One-Run-Out)")
        print("=======================================================\n")
        ensaios_a_processar = ensaios_all
    else:
        # Permite múltiplas entradas separadas por vírgula ou espaço (ex: "B09,B10" ou "B09 B10")
        val_targets = val_input.replace(',', ' ').split()
        val_targets = list(set([v for v in val_targets if v])) # Remove vazios e duplicatas
        
        ensaios_invalidos = [v for v in val_targets if v not in ensaios_all]
        
        if ensaios_invalidos:
            print(f"\n[ERRO CRÍTICO] Ensaios inválidos ou censurados detectados: {ensaios_invalidos}")
            print(f"Ensaios disponíveis para otimização: {ensaios_all}")
            sys.exit(1)
            
        print("\n=======================================================")
        print(f"   TREINAMENTO PARCIAL: Validação nos ensaios {val_targets}")
        print("=======================================================\n")
        ensaios_a_processar = val_targets

    resultados_globais = {}

    # 4. Loop de Otimização (Executa 1 ou N vezes dependendo da flag --val)
    for val_bearing in ensaios_a_processar:
        lista_validacao = [val_bearing]
        lista_treino = [b for b in ensaios_all if b != val_bearing]
        
        exp_results_dir = base_results_dir / f"val_{val_bearing}"
        exp_results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[OPERAÇÃO] Iniciando iteração: Validação isolada em {val_bearing}")
        print(f"    [+] Conjunto de Treino: {lista_treino}")
        
        print("    [+] Extração de Parâmetros Estatísticos (prevenindo vazamento de dados)")
        try:
            metadata_json_path = calcular_estatisticas_treino(
                processed_dir, 
                exp_results_dir, 
                lista_treino
            )
        except Exception as e:
            print(f"    [ERRO CRÍTICO] Falha ao extrair estatísticas para {val_bearing}.\n    Detalhe: {e}")
            continue

        print("    [+] Instanciação e Treinamento da Rede Neural Convolucional")
        try:
            erro_best = fusion_model_run(
                processed_dir=processed_dir,
                bearings_train=lista_treino,
                bearings_val=lista_validacao,
                metadata_path=metadata_json_path,
                results_dir=base_results_dir,
                epochs=50,
                batch_size=64,
                lr=0.001,
                patience=10
            )
            
            resultados_globais[val_bearing] = erro_best
            print(f"    [✔] OTIMIZAÇÃO ({val_bearing}) CONCLUÍDA. MSE Mínimo Obtido: {erro_best:.6f}")
            print("-" * 70)
            
        except RuntimeError as e:
            print(f"    [ERRO DE TEMPO DE EXECUÇÃO CUDA/PYTORCH na iteração {val_bearing}]:\n    {e}")
            continue

# 5. Relatório Final (Omitido para execuções de um único ensaio)
    if len(resultados_globais) > 1:
        print("\n=======================================================")
        print(" RESUMO DA VALIDAÇÃO CRUZADA (MSE por ensaio)")
        print("=======================================================")
        for ensaio, mse in resultados_globais.items():
            print(f" -> Validação em {ensaio}: MSE = {mse:.6f}")
        
        mse_medio = sum(resultados_globais.values()) / len(resultados_globais)
        print(f"\n >>> Erro Quadrático Médio (MSE) Global: {mse_medio:.6f}")
        print("=======================================================\n")