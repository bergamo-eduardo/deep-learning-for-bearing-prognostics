"""Pre-processing for experimental data for Deep Learning pipeline and EDA."""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import polars as pl
import scipy.io as sio
import scipy.signal as sig


OP_FEATURE_COLS = [
    "meanAbs_speed / rpm",
    "meanAbs_statLoad / N",
    "peak_dynLoad / N",
]

TRANSITION_TIMES = {
    "B01": 208, "B02": 998, "B03": 562, "B04": 452,
    "B05": 560, "B06": 12853, "B07": 6205, "B08": 1219,
    "B09": 253, "B10": 2679, "B11": 931, "B12": 154,
    "B13": 18022, "B14": 12050, "B15": 21051, "B16": 5400,
    "B17": 2323,
}

CLEAN14 = [
    "B02", "B03", "B04", "B06", "B07", "B08", "B09", "B10",
    "B11", "B12", "B13", "B14", "B15", "B17",
]
BENCHMARK17 = [f"B{i:02d}" for i in range(1, 18)]
TENSOR_FORMAT_ERROR = "Formato do tensor não match formato de entrada"


def _resolve_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Resolve one of the candidate column names in a metadata frame."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Nenhuma coluna temporal encontrada. Esperada uma de {candidates}; "
        f"colunas disponíveis: {list(df.columns)}"
    )


def _numeric_mat_sort_key(path: Path) -> int:
    match = re.search(r"M(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Nome de arquivo .mat sem índice MXXXX: {path.name}")
    return int(match.group(1))


def _chunk_index(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Nome de chunk sem índice numérico: {path.name}")
    return int(match.group(1))


def _validate_op_features(df: pd.DataFrame, source_name: str) -> None:
    missing = [column for column in OP_FEATURE_COLS if column not in df.columns]
    if missing:
        raise ValueError(f"{source_name}: colunas operacionais ausentes: {missing}")
    values = df.loc[:, OP_FEATURE_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = df.loc[:, OP_FEATURE_COLS].isna().sum().to_dict()
        raise ValueError(
            f"{source_name}: valores ausentes ou não finitos nas 3 variáveis operacionais: {bad}"
        )


def _project_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        return Path.cwd()


def preprocess_to_parquet_eda(exp: str) -> None:
    """Maintain the exploratory 4 kHz Parquet pipeline."""
    print(f"\n[EDA PIPELINE] Construindo dataset analítico do experimento {exp} (4 kHz)...")

    projeto_root = _project_root()
    base_data = projeto_root / "data"
    base_original = base_data / "raw_data" / exp
    base_processed = base_data / "processed_data" / "eda_parquet"
    temp_dir = base_data / "tmp_parquet_chunks"

    vib_data_dir = base_original / "vibrationData"
    oc_data = base_original / f"{exp}_operatingConditions.csv"
    temp_data = base_original / f"{exp}_meanTemperatures.csv"

    base_processed.mkdir(parents=True, exist_ok=True)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    fator_conversao_g = 10.0
    exp_number = int(exp.replace("B", ""))
    is_128khz = exp_number <= 9
    transition_idx = TRANSITION_TIMES.get(exp, 0)

    df_oc = pd.read_csv(oc_data)
    df_temp = pd.read_csv(temp_data)
    oc_time = _resolve_column(df_oc, ["Time", "dateTime"])
    temp_time = _resolve_column(df_temp, ["Time", "dateTime"])
    df_oc[oc_time] = pd.to_datetime(df_oc[oc_time])
    df_temp[temp_time] = pd.to_datetime(df_temp[temp_time])
    df_oc = df_oc.rename(columns={oc_time: "Time"})
    df_temp = df_temp.rename(columns={temp_time: "Time"})
    _validate_op_features(df_oc, f"{exp}_operatingConditions.csv")

    df_meta = pd.merge_asof(
        df_oc.sort_values("Time"),
        df_temp.sort_values("Time"),
        on="Time",
        direction="backward",
    )
    df_meta["metaTime"] = (df_meta["Time"] - df_meta["Time"].min()).dt.total_seconds()
    _validate_op_features(df_meta, "merged metadata")

    matlab_files = sorted(
        [file for file in vib_data_dir.iterdir() if file.suffix == ".mat"],
        key=_numeric_mat_sort_key,
    )
    if not matlab_files:
        print(f"[ERRO] Nenhum arquivo .mat localizado em: {vib_data_dir}")
        sys.exit(1)

    last_meta_row = df_meta.iloc[-1]
    last_mat_data = sio.loadmat(matlab_files[-1])
    last_meas_time = last_mat_data["measTime"].flatten()
    global_max_time_s = float(last_meta_row["metaTime"] + last_meas_time[-1])
    del last_mat_data, last_meas_time
    gc.collect()

    for i, file_path in enumerate(matlab_files):
        match = re.search(r"M(\d+)", file_path.stem)
        file_idx = int(match.group(1)) if match else i + 1
        if file_idx < transition_idx:
            continue

        mat_data = sio.loadmat(file_path)
        acc_a = mat_data["accHorizRear_A"].flatten() * fator_conversao_g
        acc_c = mat_data["accHorizFrontal_C"].flatten() * fator_conversao_g
        meas_time = mat_data["measTime"].flatten()

        if is_128khz:
            acc_a = sig.decimate(sig.decimate(acc_a, q=8, ftype="fir"), q=4, ftype="fir")
            acc_c = sig.decimate(sig.decimate(acc_c, q=8, ftype="fir"), q=4, ftype="fir")
            meas_time = meas_time[::32]
        else:
            acc_a = sig.decimate(sig.decimate(acc_a, q=8, ftype="fir"), q=2, ftype="fir")
            acc_c = sig.decimate(sig.decimate(acc_c, q=8, ftype="fir"), q=2, ftype="fir")
            meas_time = meas_time[::16]

        acc_a = sig.detrend(acc_a, type="constant")
        acc_c = sig.detrend(acc_c, type="constant")
        common_length = min(len(acc_a), len(acc_c), len(meas_time))
        acc_a = acc_a[:common_length]
        acc_c = acc_c[:common_length]
        meas_time = meas_time[:common_length]
        meta_index = min(i, len(df_meta) - 1)
        meta_row = df_meta.iloc[meta_index]
        absolute_time_array = meta_row["metaTime"] + meas_time
        rul_array = global_max_time_s - absolute_time_array

        df_chunk = pl.DataFrame({
            "Rul_s": rul_array,
            "setDynLoad_N": np.full(len(meas_time), meta_row["setDynLoad / N"], dtype=np.float64),
            "setStatLoad_N": np.full(len(meas_time), meta_row["setStatLoad / N"], dtype=np.float64),
            "setSpeed_rpm": np.full(len(meas_time), meta_row["setSpeed / rpm"], dtype=np.float64),
            "peakDynLoad_N": np.full(len(meas_time), meta_row["peak_dynLoad / N"], dtype=np.float64),
            "StatLoad_N": np.full(len(meas_time), meta_row["meanAbs_statLoad / N"], dtype=np.float64),
            "Speed_rpm": np.full(len(meas_time), meta_row["meanAbs_speed / rpm"], dtype=np.float64),
            "TempT1_C": np.full(len(meas_time), meta_row["Mean Abs. Temp. T1 / °C"], dtype=np.float64),
            "TempT2_C": np.full(len(meas_time), meta_row["Mean Abs. Temp. T2 / °C"], dtype=np.float64),
            "RoomTemp_C": np.full(len(meas_time), meta_row["Mean Room Temp. / °C"], dtype=np.float64),
            "AccelA_g": acc_a,
            "AccelC_g": acc_c,
        })

        chunk_path = temp_dir / f"chunk_{i:04d}.parquet"
        df_chunk.write_parquet(chunk_path)
        del mat_data, acc_a, acc_c, meas_time, absolute_time_array, rul_array, df_chunk
        gc.collect()

    output_file = base_processed / f"eda_{exp}_4kHz.parquet"
    try:
        lazy_dataset = pl.scan_parquet(temp_dir / "*.parquet")
        lazy_dataset.sink_parquet(output_file)
        print(f"[SUCESSO] Dataset analítico (EDA) consolidado em: {output_file}")
    except Exception as exc:
        print(f"[AVISO] Nenhum dado gerado para {exp} após o filtro de transição: {exc}")
    shutil.rmtree(temp_dir, ignore_errors=True)


def preprocess_to_npz(
    exp: str,
    window_size: int = 4096,
    seq_length: int = 10,
    f_amostragem_kHz: int = 32,
) -> None:
    """Generate schema-v2 out-of-core vibration tensors for one bearing."""
    print(f"\n[TENSOR PIPELINE] Processando {exp} em {f_amostragem_kHz} kHz...")

    projeto_root = _project_root()
    base_original = projeto_root / "data" / "raw_data" / exp
    base_processed = projeto_root / "data" / "processed_data"
    tensors_dir = base_processed / f"{exp}_tensors"
    vib_data_dir = base_original / "vibrationData"
    oc_data = base_original / f"{exp}_operatingConditions.csv"
    temp_data = base_original / f"{exp}_meanTemperatures.csv"

    if not oc_data.exists() or not temp_data.exists():
        raise FileNotFoundError(f"Arquivos CSV operacionais ausentes em: {base_original}")
    if exp not in TRANSITION_TIMES:
        raise ValueError(f"Experimento sem TRANSITION_TIMES: {exp}")

    base_processed.mkdir(parents=True, exist_ok=True)
    if tensors_dir.exists():
        shutil.rmtree(tensors_dir)
    tensors_dir.mkdir(parents=True, exist_ok=True)

    df_oc = pd.read_csv(oc_data)
    df_temp = pd.read_csv(temp_data)
    oc_time = _resolve_column(df_oc, ["Time", "dateTime"])
    temp_time = _resolve_column(df_temp, ["Time", "dateTime"])
    df_oc[oc_time] = pd.to_datetime(df_oc[oc_time])
    df_temp[temp_time] = pd.to_datetime(df_temp[temp_time])
    df_oc = df_oc.rename(columns={oc_time: "Time"})
    df_temp = df_temp.rename(columns={temp_time: "Time"})
    _validate_op_features(df_oc, f"{exp}_operatingConditions.csv")

    df_meta = pd.merge_asof(
        df_oc.sort_values("Time"),
        df_temp.sort_values("Time"),
        on="Time",
        direction="backward",
    )
    df_meta["metaTime"] = (df_meta["Time"] - df_meta["Time"].min()).dt.total_seconds()
    _validate_op_features(df_meta, f"{exp} metadata")

    matlab_files = sorted(
        [file for file in vib_data_dir.iterdir() if file.suffix == ".mat"],
        key=_numeric_mat_sort_key,
    )
    if not matlab_files:
        raise FileNotFoundError(f"Diretório de vibração vazio: {vib_data_dir}")

    file_indices = [_numeric_mat_sort_key(file) for file in matlab_files]
    total_cycles = max(file_indices)
    tt_idx = TRANSITION_TIMES[exp]
    if total_cycles <= tt_idx:
        raise ValueError(
            f"{exp}: total_cycles={total_cycles} deve ser maior que "
            f"tt_idx={tt_idx}"
        )

    last_meta_row = df_meta.iloc[-1]
    last_mat_data = sio.loadmat(matlab_files[-1])
    last_meas_time = last_mat_data["measTime"].flatten()
    global_max_time_s = float(last_meta_row["metaTime"] + last_meas_time[-1])
    del last_mat_data, last_meas_time
    gc.collect()

    is_128khz = int(exp.replace("B", "")) <= 9
    fator_q = 128 // f_amostragem_kHz if is_128khz else 64 // f_amostragem_kHz
    if fator_q < 1:
        raise ValueError("f_amostragem_kHz excede a resolução de decimação disponível")

    total_sequences_generated = 0
    for i, file_path in enumerate(matlab_files):
        file_idx = _numeric_mat_sort_key(file_path)
        if file_idx < tt_idx:
            continue

        mat_data = sio.loadmat(file_path)
        acc_a = mat_data["accHorizRear_A"].flatten() * 10.0
        acc_c = mat_data["accHorizFrontal_C"].flatten() * 10.0
        meas_time = mat_data["measTime"].flatten()
        acc_a = sig.decimate(acc_a, q=fator_q, ftype="fir")
        acc_c = sig.decimate(acc_c, q=fator_q, ftype="fir")
        meas_time = meas_time[::fator_q]
        acc_a = sig.detrend(acc_a, type="constant")
        acc_c = sig.detrend(acc_c, type="constant")
        common_length = min(len(acc_a), len(acc_c), len(meas_time))
        acc_a = acc_a[:common_length]
        acc_c = acc_c[:common_length]
        meas_time = meas_time[:common_length]

        meta_index = min(i, len(df_meta) - 1)
        meta_row = df_meta.iloc[meta_index]
        absolute_time_array = meta_row["metaTime"] + meas_time
        y_rul_s = global_max_time_s - absolute_time_array
        y_pct_cycle = float(np.clip(
            (total_cycles - file_idx) / (total_cycles - tt_idx),
            0.0,
            1.0,
        ))

        vib_matrix = np.column_stack((acc_a, acc_c))
        num_windows = len(vib_matrix) // window_size
        if num_windows < seq_length:
            del mat_data, acc_a, acc_c, meas_time, vib_matrix, absolute_time_array, y_rul_s
            gc.collect()
            continue

        vib_trunc = vib_matrix[:num_windows * window_size, :]
        rul_trunc = y_rul_s[:num_windows * window_size]
        X_vib_3d = vib_trunc.reshape((num_windows, window_size, 2))
        X_vib_3d = np.transpose(X_vib_3d, (0, 2, 1)).astype(np.float32)
        X_vib_seq_view = np.lib.stride_tricks.sliding_window_view(
            X_vib_3d,
            window_shape=seq_length,
            axis=0,
        )
        X_vib_seq = np.transpose(X_vib_seq_view, (0, 3, 1, 2)).astype(np.float32)

        rul_2d = rul_trunc.reshape((num_windows, window_size))
        y_rul_s = rul_2d[:, -1]
        op_vec = np.array([meta_row[column] for column in OP_FEATURE_COLS], dtype=np.float32)
        X_op_seq = np.repeat(op_vec[None, :], len(X_vib_seq), axis=0)
        y_pct_seq = np.full(len(X_vib_seq), y_pct_cycle, dtype=np.float32)
        y_s_seq = y_rul_s[seq_length - 1:].astype(np.float32)
        file_idx_seq = np.full(len(X_vib_seq), file_idx, dtype=np.int32)

        if not (
            np.isfinite(X_vib_seq).all()
            and np.isfinite(X_op_seq).all()
            and np.isfinite(y_pct_seq).all()
            and np.isfinite(y_s_seq).all()
        ):
            raise ValueError(TENSOR_FORMAT_ERROR)
        if np.any((y_pct_seq < 0.0) | (y_pct_seq > 1.0)):
            raise ValueError(TENSOR_FORMAT_ERROR)

        chunk_path = tensors_dir / f"tensor_chunk_{file_idx:05d}.npz"
        np.savez_compressed(
            chunk_path,
            X_vib=X_vib_seq,
            X_op=X_op_seq,
            y_pct=y_pct_seq,
            y_s=y_s_seq,
            file_idx=file_idx_seq,
            op_feature_names=np.asarray(OP_FEATURE_COLS, dtype=np.str_),
        )
        total_sequences_generated += len(X_vib_seq)
        del mat_data, acc_a, acc_c, meas_time, vib_matrix, vib_trunc, rul_trunc
        del X_vib_3d, X_vib_seq_view, X_vib_seq, rul_2d, y_rul_s
        del op_vec, X_op_seq, y_pct_seq, y_s_seq, file_idx_seq
        gc.collect()

    print(
        f"[SUCESSO] Tensores salvos em {tensors_dir}: "
        f"{total_sequences_generated} sequências."
    )


def calcular_estatisticas_treino(
    processed_dir: Path,
    target_dir: Path,
    bearings_train: List[str],
) -> Path:
    """Calculate global operational z-score statistics with O(1) memory."""
    total_count = 0
    sum_x = np.zeros(3, dtype=np.float64)
    sum_x2 = np.zeros(3, dtype=np.float64)

    for exp in bearings_train:
        tensor_path = processed_dir / f"{exp}_tensors"
        files = sorted(tensor_path.glob("tensor_chunk_*.npz"), key=_chunk_index)
        for file_path in files:
            with np.load(file_path, allow_pickle=False) as data:
                if "X_op" not in data.files:
                    raise ValueError(TENSOR_FORMAT_ERROR)
                X_op = data["X_op"]
                if X_op.ndim != 2 or X_op.shape[1] != 3:
                    raise ValueError(TENSOR_FORMAT_ERROR)
                if not np.isfinite(X_op).all():
                    raise ValueError(TENSOR_FORMAT_ERROR)
                total_count += X_op.shape[0]
                sum_x += X_op.astype(np.float64).sum(axis=0)
                sum_x2 += np.square(X_op.astype(np.float64)).sum(axis=0)

    if total_count == 0:
        raise RuntimeError(
            f"Falha de leitura: nenhum tensor X_op localizado para {bearings_train}"
        )

    op_mean = sum_x / total_count
    op_std = np.sqrt(np.maximum(sum_x2 / total_count - op_mean**2, 0.0)) + 1e-8
    metadata = {
        "schema_version": 2,
        "target_mode": "rul_pct_post_tt",
        "op_feature_names": OP_FEATURE_COLS,
        "num_op_features": 3,
        "op_mean": op_mean.astype(np.float32).tolist(),
        "op_std": op_std.astype(np.float32).tolist(),
    }

    target_dir.mkdir(parents=True, exist_ok=True)
    output_json = target_dir / "metadata_normalization.json"
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)
    return output_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-processamento de dados experimentais")
    parser.add_argument("--split", choices=["clean14", "benchmark17"])
    parser.add_argument("--exp", type=str)
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--seq-length", type=int, default=10)
    parser.add_argument("--sample-rate-khz", type=int, default=32)
    args = parser.parse_args()

    if args.exp:
        experiments = [item.strip().upper() for item in args.exp.split(",") if item.strip()]
    elif args.split == "benchmark17":
        experiments = BENCHMARK17
    else:
        experiments = CLEAN14

    for experiment in experiments:
        preprocess_to_npz(
            experiment,
            window_size=args.window_size,
            seq_length=args.seq_length,
            f_amostragem_kHz=args.sample_rate_khz,
        )
