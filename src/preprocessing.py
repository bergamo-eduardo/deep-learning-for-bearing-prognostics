"""
Módulo de Pré-processamento Digital de Sinais e Integração de Metadados
Trabalho de Conclusão de Curso - Estimativa de RUL via CNN
Autor: Eduardo Kanadani Bergamo
Orientador: Prof. Dr. Pedro Fernando Poveda
"""

import polars as pl
import pandas as pd
import numpy as np
import scipy.io as sio
import scipy.signal as sig
from pathlib import Path
import shutil
import gc
import sys

def processar_experimento(exp: str):
    print(f"Iniciando processamento analítico do experimento {exp}...")
    
    # --- 1. Resolução Dinâmica de Diretórios ---
    # __file__ localiza o script atual (src/preprocessing.py) e parents[1] aponta para a raiz do repositório
    try:
        projeto_root = Path(__file__).resolve().parents[1]
    except NameError:
        # Fallback caso executado em ambiente interativo não-padrão
        projeto_root = Path.cwd()

    base_data = projeto_root / 'data'
    base_original = base_data / 'raw_data' / exp
    base_processed = base_data / 'processed_data'
    temp_dir = base_data / 'tmp_parquet_chunks'

    vib_data_dir = base_original / 'vibrationData'
    oc_data = base_original / f'{exp}_operatingConditions.csv'
    temp_data = base_original / f'{exp}_meanTemperatures.csv'

    # Criação e limpeza de diretórios
    base_processed.mkdir(parents=True, exist_ok=True)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Parâmetros Físicos e Metodológicos ---
    # Fator de conversão da sensibilidade do acelerômetro (assumindo 10g/V -> fator 10)
    FATOR_CONVERSAO_G = 10.0
    
    # Avaliação da necessidade de filtro anti-aliasing e subamostragem (decimação)
    # Segundo a documentação da base, B01 a B09 = 128 kHz. B10 a B17 = 64 kHz.
    exp_number = int(exp.replace('B', ''))
    is_128kHz = exp_number <= 9

    # --- 3. Processamento Relacional dos Metadados (Pandas) ---
    print("Sincronizando variáveis de controle e variáveis de processo termodinâmico...")
    df_oc = pd.read_csv(oc_data)
    df_temp = pd.read_csv(temp_data)

    df_oc['Time'] = pd.to_datetime(df_oc['Time'])
    df_temp['Time'] = pd.to_datetime(df_temp['Time'])

    # Mesclagem das medições pelo tempo mais próximo (nearest)
    df_meta = pd.merge_asof(
        df_oc.sort_values('Time'),
        df_temp.sort_values('Time'),
        on='Time',
        direction='nearest'
    )

    # Criação de um vetor de tempo macro (em segundos) desde o início do experimento
    df_meta['metaTime'] = (df_meta['Time'] - df_meta['Time'].min()).dt.total_seconds()

    # --- 4. Obtenção Determinística do Tempo de Falha (Max Time) ---
    # Fundamental para calcular a RUL sem esgotar a RAM em operações de janela globais
    matlab_files = sorted([f for f in vib_data_dir.iterdir() if f.suffix == '.mat'])
    if not matlab_files:
        print(f"[ERRO] Nenhum arquivo .mat encontrado no diretório: {vib_data_dir}")
        sys.exit(1)

    last_meta_row = df_meta.iloc[-1]
    last_mat_data = sio.loadmat(matlab_files[-1])
    last_meas_time = last_mat_data['measTime'].flatten()
    
    # Ajusta o vetor de tempo se houver decimação
    if is_128kHz:
        last_meas_time = last_meas_time[::2]
        
    global_max_time_s = last_meta_row['metaTime'] + last_meas_time[-1]

    # Limpeza da RAM
    del last_mat_data, last_meas_time
    gc.collect()

    # --- 5. Processamento Digital de Sinais (Chunking) ---
    print("Aplicando processamento digital de sinais em lotes (Out-of-Core)...")
    
    for i, file_path in enumerate(matlab_files):
        # Carregamento estrito à memória do arquivo em vigência
        mat_data = sio.loadmat(file_path)
        
        acc_a = mat_data['accHorizRear_A'].flatten() * FATOR_CONVERSAO_G
        acc_c = mat_data['accHorizFrontal_C'].flatten() * FATOR_CONVERSAO_G
        meas_time = mat_data['measTime'].flatten()

        # Decimação metodologicamente correta (Filtro FIR Anti-aliasing q=2)
        if is_128kHz:
            acc_a = sig.decimate(acc_a, q=2, ftype='fir')
            acc_c = sig.decimate(acc_c, q=2, ftype='fir')
            meas_time = meas_time[::2]
            
        # Sincronização temporal do bloco (chunk) de alta frequência com a linha de metadados
        meta_index = min(i, len(df_meta) - 1)
        meta_row = df_meta.iloc[meta_index]
        
        # O tempo absoluto da amostra é o tempo do metadado + o tempo decorrido no sinal .mat
        absolute_time_array = meta_row['metaTime'] + meas_time
        rul_array = global_max_time_s - absolute_time_array

        # Estruturação tabular via Polars para alta eficiência de exportação (Parquet)
        df_chunk = pl.DataFrame({
            'Rul_s': rul_array,
            
            # --- Variáveis de Controle (Setpoints impostos ao sistema) ---
            'setDynLoad_N': np.full(len(meas_time), meta_row['setDynLoad / N'], dtype=np.float64),
            'setStatLoad_N': np.full(len(meas_time), meta_row['setStatLoad / N'], dtype=np.float64),
            'setSpeed_rpm': np.full(len(meas_time), meta_row['setSpeed / rpm'], dtype=np.float64),
            
            # --- Variáveis de Processo (Medições reais obtidas de sensores) ---
            'peakDynLoad_N': np.full(len(meas_time), meta_row['peak_dynLoad / N'], dtype=np.float64),
            'StatLoad_N': np.full(len(meas_time), meta_row['meanAbs_statLoad / N'], dtype=np.float64),
            'Speed_rpm': np.full(len(meas_time), meta_row['meanAbs_speed / rpm'], dtype=np.float64),
            'TempT1_C': np.full(len(meas_time), meta_row['Mean Abs. Temp. T1 / °C'], dtype=np.float64),
            'TempT2_C': np.full(len(meas_time), meta_row['Mean Abs. Temp. T2 / °C'], dtype=np.float64),
            'RoomTemp_C': np.full(len(meas_time), meta_row['Mean Room Temp. / °C'], dtype=np.float64),
            
            # Sinais cinemáticos dinâmicos
            'AccelA_g': acc_a,
            'AccelC_g': acc_c,
        })
        
        # Exporta o lote processado para o diretório temporário
        chunk_path = temp_dir / f"chunk_{i:04d}.parquet"
        df_chunk.write_parquet(chunk_path)
        
        # Invocação explícita do Garbage Collector
        del mat_data, acc_a, acc_c, meas_time, absolute_time_array, rul_array, df_chunk
        gc.collect()

    # --- 6. Consolidação e Escrita Preguiçosa (Lazy API Streaming) ---
    print("Consolidando frações de dados em arquitetura Lazy API...")
    output_file = base_processed / f"dataset_processado_{exp}_64kHz.parquet"
    
    # scan_parquet não aloca os dados na RAM, apenas monta o plano de execução
    lazy_dataset = pl.scan_parquet(temp_dir / "*.parquet")
    
    # sink_parquet iterará sobre o disco em fluxo contínuo
    lazy_dataset.sink_parquet(output_file)

    # Limpeza metodológica pós-execução
    shutil.rmtree(temp_dir)
    print(f"Processamento concluído com êxito! Arquivo final gerado: {output_file}")

if __name__ == '__main__':
    exp_batch = [
        'B01',
        'B02',
        'B03',
        'B04',
        'B05',
        'B06',
        'B07',
        'B08',
        'B09',
        'B10',
        'B11',
        'B12',
        'B13',
        'B14',
        'B15',
        'B16',
        'B17'
        ]
    for exp_atual in exp_batch:
        processar_experimento(exp_atual)