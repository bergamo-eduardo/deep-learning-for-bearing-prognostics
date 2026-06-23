"""
Módulo Unificado de Integração de Metadados e Pré-processamento
Trabalho de Conclusão de Curso - Estimativa de RUL via Deep Learning
Foco: Tensores 4D para CNN-LSTM (LORO) e Extração Parquet (4kHz) para EDA

Autor: Eduardo Kanadani Bergamo
Orientador: Prof. Dr. Pedro Fernando Poveda
"""

import gc
import glob
import json
import shutil
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import polars as pl
import scipy.io as sio
import scipy.signal as sig


def preprocess_to_parquet_eda(exp: str) -> None:
    """
    Processa os sinais brutos e os converte em um conjunto de dados tabular único
    em formato Parquet para Análise Exploratória de Dados (EDA).
    Aplica decimação em cascata para atingir uma taxa de amostragem efetiva de 4 kHz.
    """
    print(f"\n[EDA PIPELINE] Construindo dataset analítico do experimento {exp} (4 kHz)...")
    
    # 1. Resolução de Diretórios
    try:
        projeto_root = Path(__file__).resolve().parents[1]
    except NameError:
        projeto_root = Path.cwd()

    base_data = projeto_root / 'data'
    base_original = base_data / 'raw_data' / exp
    base_processed = base_data / 'processed_data' / 'eda_parquet'
    temp_dir = base_data / 'tmp_parquet_chunks'

    vib_data_dir = base_original / 'vibrationData'
    oc_data = base_original / f'{exp}_operatingConditions.csv'
    temp_data = base_original / f'{exp}_meanTemperatures.csv'

    base_processed.mkdir(parents=True, exist_ok=True)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    FATOR_CONVERSAO_G = 10.0
    exp_number = int(exp.replace('B', ''))
    is_128kHz = exp_number <= 9

    # 2. Sincronização Relacional (As-Of Join)
    df_oc = pd.read_csv(oc_data)
    df_temp = pd.read_csv(temp_data)
    df_oc['Time'] = pd.to_datetime(df_oc['Time'])
    df_temp['Time'] = pd.to_datetime(df_temp['Time'])

    df_meta = pd.merge_asof(
        df_oc.sort_values('Time'),
        df_temp.sort_values('Time'),
        on='Time',
        direction='backward'
    )
    df_meta['metaTime'] = (df_meta['Time'] - df_meta['Time'].min()).dt.total_seconds()

    # 3. Determinação do Tempo Máximo (Vida Útil)
    matlab_files = sorted([f for f in vib_data_dir.iterdir() if f.suffix == '.mat'])
    if not matlab_files:
        print(f"[ERRO] Nenhum arquivo .mat localizado em: {vib_data_dir}")
        sys.exit(1)

    last_meta_row = df_meta.iloc[-1]
    last_mat_data = sio.loadmat(matlab_files[-1])
    last_meas_time = last_mat_data['measTime'].flatten()
    
    global_max_time_s = last_meta_row['metaTime'] + last_meas_time[-1]

    del last_mat_data, last_meas_time
    gc.collect()

    # 4. Processamento Digital em Lotes (Out-of-Core)
    for i, file_path in enumerate(matlab_files):
        mat_data = sio.loadmat(file_path)
        
        acc_a = mat_data['accHorizRear_A'].flatten() * FATOR_CONVERSAO_G
        acc_c = mat_data['accHorizFrontal_C'].flatten() * FATOR_CONVERSAO_G
        meas_time = mat_data['measTime'].flatten()

        # Decimação Estratégica em Cascata para preservação da banda de transição do filtro FIR (Alvo: 4 kHz)
        if is_128kHz:
            # Redução total: Fator 32 (8 * 4)
            acc_a = sig.decimate(sig.decimate(acc_a, q=8, ftype='fir'), q=4, ftype='fir')
            acc_c = sig.decimate(sig.decimate(acc_c, q=8, ftype='fir'), q=4, ftype='fir')
            meas_time = meas_time[::32]
        else:
            # Redução total: Fator 16 (8 * 2)
            acc_a = sig.decimate(sig.decimate(acc_a, q=8, ftype='fir'), q=2, ftype='fir')
            acc_c = sig.decimate(sig.decimate(acc_c, q=8, ftype='fir'), q=2, ftype='fir')
            meas_time = meas_time[::16]
            
        # Supressão de componente contínua (DC offset)
        acc_a = sig.detrend(acc_a, type='constant')
        acc_c = sig.detrend(acc_c, type='constant')

        meta_index = min(i, len(df_meta) - 1)
        meta_row = df_meta.iloc[meta_index]
        
        absolute_time_array = meta_row['metaTime'] + meas_time
        rul_array = global_max_time_s - absolute_time_array

        # Inserção Tabular Otimizada
        df_chunk = pl.DataFrame({
            'Rul_s': rul_array,
            'setDynLoad_N': np.full(len(meas_time), meta_row['setDynLoad / N'], dtype=np.float64),
            'setStatLoad_N': np.full(len(meas_time), meta_row['setStatLoad / N'], dtype=np.float64),
            'setSpeed_rpm': np.full(len(meas_time), meta_row['setSpeed / rpm'], dtype=np.float64),
            'peakDynLoad_N': np.full(len(meas_time), meta_row['peak_dynLoad / N'], dtype=np.float64),
            'StatLoad_N': np.full(len(meas_time), meta_row['meanAbs_statLoad / N'], dtype=np.float64),
            'Speed_rpm': np.full(len(meas_time), meta_row['meanAbs_speed / rpm'], dtype=np.float64),
            'TempT1_C': np.full(len(meas_time), meta_row['Mean Abs. Temp. T1 / °C'], dtype=np.float64),
            'TempT2_C': np.full(len(meas_time), meta_row['Mean Abs. Temp. T2 / °C'], dtype=np.float64),
            'RoomTemp_C': np.full(len(meas_time), meta_row['Mean Room Temp. / °C'], dtype=np.float64),
            'AccelA_g': acc_a,
            'AccelC_g': acc_c,
        })
        
        chunk_path = temp_dir / f"chunk_{i:04d}.parquet"
        df_chunk.write_parquet(chunk_path)
        
        del mat_data, acc_a, acc_c, meas_time, absolute_time_array, rul_array, df_chunk
        gc.collect()

    # 5. Consolidação Unificada
    output_file = base_processed / f"eda_{exp}_4kHz.parquet"
    lazy_dataset = pl.scan_parquet(temp_dir / "*.parquet")
    lazy_dataset.sink_parquet(output_file)

    shutil.rmtree(temp_dir)
    print(f"[SUCESSO] Dataset analítico (EDA) consolidado e armazenado em: {output_file}")

def preprocess_to_npz(exp: str, window_size: int, seq_length: int, f_amostragem_kHz: int) -> None:
    """
    Realiza o pré-processamento dos dados brutos experimentais, aplicando sincronização
    relacional, decimação dinâmica e estruturação em tensores quadridimensionais
    para o treinamento sequencial. Alvo de amostragem estabelecido em 32 kHz.
    
    Parâmetros:
    -----------
    exp : str
        Identificador alfanumérico do ensaio (ex.: 'B01').
    window_size : int
        Número de amostras pontuais (discretas) que compõem cada janela.
    seq_length : int
        Número de janelas concatenadas para formar a trajetória de degradação.
    """
    print(f"\n[TENSOR PIPELINE] Iniciando conversão matricial sequencial do experimento {exp} ({f_amostragem_kHz} kHz)...")
    
    try:
        projeto_root = Path(__file__).resolve().parents[1]
    except NameError:
        projeto_root = Path.cwd()

    base_data = projeto_root / 'data'
    base_original = base_data / 'raw_data' / exp
    base_processed = base_data / 'processed_data'
    tensors_dir = base_processed / f'{exp}_tensors'
    
    vib_data_dir = base_original / 'vibrationData'
    oc_data = base_original / f'{exp}_operatingConditions.csv'
    temp_data = base_original / f'{exp}_meanTemperatures.csv'

    base_processed.mkdir(parents=True, exist_ok=True)
    if tensors_dir.exists():
        shutil.rmtree(tensors_dir)
    tensors_dir.mkdir(parents=True, exist_ok=True)

    FATOR_CONVERSAO_G = 10.0
    exp_number = int(exp.replace('B', ''))
    is_128kHz = exp_number <= 9

    if not (oc_data.exists() and temp_data.exists()):
        print(f"[ERRO] Arquivos de variáveis operacionais (CSV) ausentes em: {base_original}")
        sys.exit(1)
        
    df_oc = pd.read_csv(oc_data)
    df_temp = pd.read_csv(temp_data)

    df_oc['Time'] = pd.to_datetime(df_oc['Time'])
    df_temp['Time'] = pd.to_datetime(df_temp['Time'])

    df_meta = pd.merge_asof(
        df_oc.sort_values('Time'),
        df_temp.sort_values('Time'),
        on='Time',
        direction='backward'
    )
    df_meta['metaTime'] = (df_meta['Time'] - df_meta['Time'].min()).dt.total_seconds()

    matlab_files = sorted([f for f in vib_data_dir.iterdir() if f.suffix == '.mat'])
    if not matlab_files:
        print(f"[ERRO] Diretório de dados de vibração vazio: {vib_data_dir}")
        sys.exit(1)

    last_meta_row = df_meta.iloc[-1]
    last_mat_data = sio.loadmat(matlab_files[-1])
    last_meas_time = last_mat_data['measTime'].flatten()
    
    global_max_time_s = last_meta_row['metaTime'] + last_meas_time[-1]

    del last_mat_data, last_meas_time
    gc.collect()

    total_sequences_generated = 0
    
    for i, file_path in enumerate(matlab_files):
        mat_data = sio.loadmat(file_path)
        
        acc_a = mat_data['accHorizRear_A'].flatten() * FATOR_CONVERSAO_G
        acc_c = mat_data['accHorizFrontal_C'].flatten() * FATOR_CONVERSAO_G
        meas_time = mat_data['measTime'].flatten()

        # Determinação do fator de subamostragem baseado na frequência original
        # Assegurando taxa final idêntica (32 kHz) para todos os ensaios
        fator_q = 128//f_amostragem_kHz if is_128kHz else 64//f_amostragem_kHz

        # Aplicação de decimação e filtro FIR anti-aliasing
        acc_a = sig.decimate(acc_a, q=fator_q, ftype='fir')
        acc_c = sig.decimate(acc_c, q=fator_q, ftype='fir')
        meas_time = meas_time[::fator_q]
        
        # Centralização da média da forma de onda em zero
        acc_a = sig.detrend(acc_a, type='constant')
        acc_c = sig.detrend(acc_c, type='constant')
            
        meta_index = min(i, len(df_meta) - 1)
        meta_row = df_meta.iloc[meta_index]
        
        absolute_time_array = meta_row['metaTime'] + meas_time
        rul_array = global_max_time_s - absolute_time_array

        X_matrix = np.column_stack((
            acc_a,
            acc_c,
            np.full(len(meas_time), meta_row['meanAbs_speed / rpm']), 
            np.full(len(meas_time), meta_row['meanAbs_statLoad / N']) 
        ))
        
        num_windows = len(X_matrix) // window_size
        if num_windows < seq_length:
            continue
            
        X_trunc = X_matrix[:num_windows * window_size, :]
        rul_trunc = rul_array[:num_windows * window_size]
        
        # Segmentação das amostras em blocos bidimensionais (janelas)
        X_3d = X_trunc.reshape((num_windows, window_size, X_matrix.shape[1]))
        X_3d = np.transpose(X_3d, (0, 2, 1))
        
        rul_2d = rul_trunc.reshape((num_windows, window_size))
        y_rul = rul_2d[:, -1]
        
        # Formação das matrizes sequenciais (Sliding Window para histórico LSTM)
        X_seq_view = np.lib.stride_tricks.sliding_window_view(X_3d, window_shape=seq_length, axis=0)
        X_seq = np.transpose(X_seq_view, (0, 3, 1, 2))
        
        y_seq = y_rul[seq_length - 1:]
        
        chunk_path = tensors_dir / f"tensor_chunk_{i:04d}.npz"
        np.savez_compressed(
            chunk_path, 
            X=X_seq.astype(np.float32), 
            y=y_seq.astype(np.float32)
        )
        total_sequences_generated += len(X_seq)
        
        del mat_data, acc_a, acc_c, meas_time, X_matrix, X_trunc, rul_trunc, X_3d, X_seq_view, X_seq
        gc.collect()

    print(f"[SUCESSO] Operação estrutural concluída. Quantidade de sequências geradas: {total_sequences_generated}.")

def calcular_estatisticas_treino(processed_dir: Path, target_dir: Path, bearings_train: List[str]) -> Path:
    """
    Varre os arquivos gerados correspondentes à partição de treinamento estipulada
    para extração dos parâmetros estatísticos globais. A rotina assegura a invariabilidade
    das escalas gravando o metadado no diretório isolado da respectiva iteração LORO.
    """
    print(f"\n[ESTATÍSTICA] Computando métricas de normalização global sobre a amostragem: {bearings_train}")
    op_accumulated = []
    y_max_global = 0.0

    for exp in bearings_train:
        tensor_path = processed_dir / f"{exp}_tensors"
        files = sorted(glob.glob(str(tensor_path / "tensor_chunk_*.npz")))
        
        for file in files:
            with np.load(file) as data:
                X = data["X"]
                y = data["y"]
                # Extração isolada dos tensores operacionais
                op_features = X[:, :, 2:4, 0].reshape(-1, 2)
                op_accumulated.append(op_features)
                
                local_y_max = float(np.max(y))
                if local_y_max > y_max_global:
                    y_max_global = local_y_max

    if not op_accumulated:
        raise RuntimeError(f"Falha de leitura. Os tensores não foram localizados para os ensaios: {bearings_train}.")

    op_matrix = np.vstack(op_accumulated)
    op_mean = np.mean(op_matrix, axis=0).tolist()
    op_std = (np.std(op_matrix, axis=0) + 1e-8).tolist()

    metadata = {
        "op_mean": op_mean,
        "op_std": op_std,
        "y_max": y_max_global
    }

    # Garante a existência do diretório específico da iteração antes de salvar
    target_dir.mkdir(parents=True, exist_ok=True)
    output_json = target_dir / "metadata_normalization.json"
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[SUCESSO] Parâmetros estatísticos isolados e salvos em: {output_json}")
    return output_json


if __name__ == '__main__':
    # Ensaios selecionados
    exp_batch = [
        "B02",
        "B03",
        "B04",
        "B07",
        "B08",
        "B09",
        "B10",
        "B11",
        "B12",
        "B14",
        "B17"
    ]
    

    
    # Executar apenas se houver necessidade de nova exploração no notebook Pandas
    for experimento in exp_batch:
        preprocess_to_parquet_eda(experimento)
        
    # # Construção rigorosa de matrizes sequenciais para alimentação computacional
    # for experimento in exp_batch:
    #     preprocess_to_npz(experimento, window_size=4096, seq_length=10, f_amostragem_kHz=32)