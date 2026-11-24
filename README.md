# Remaining Useful Life Estimation for Bearings Using Deep Learning

This repository documents the technical development of a Mechanical Engineering capstone project on applying deep learning to bearing prognostics. The project estimates the *Remaining Useful Life* (RUL) of bearings by combining high-frequency vibration measurements with operating-condition data from the Paderborn University bearing experiments.

---

## 1. Technical Dependencies

Install the dependencies listed in [`requirements.txt`](requirements.txt). The minimum versions below match the package requirements.

### 1.1 Data Handling and Performance
* **Polars (>=0.20.0):** Used by the exploratory Parquet pipeline to consolidate large datasets with lazy, out-of-core processing.
* **Pandas (>=2.0.0):** Used to align operating-condition and temperature metadata by timestamp and to summarize model predictions.
* **NumPy (>=1.24.0):** Used for signal arrays, tensor preparation, and RUL targets.

### 1.2 Digital Signal Processing (DSP)
* **SciPy (>=1.10.0):** Loads MATLAB `.mat` measurements and applies FIR decimation and detrending to vibration signals.

### 1.3 Storage and Deep Learning
* **PyArrow (>=14.0.0):** Provides Parquet support for the exploratory data pipeline.
* **PyTorch (>=2.0.0):** Implements the CNN–LSTM model and training loop.
* **tqdm (>=4.65.0):** Displays training progress.

### 1.4 Visualization and Diagnostics
* **Matplotlib (>=3.7.0)** and **Seaborn (>=0.12.0):** Used by the exploratory notebooks to visualize vibration signals and operating trends.

### 1.5 Environment Setup
From the repository root, install the dependencies and create any missing project directories:

```bash
python -m pip install -r requirements.txt
python project_setup.py
```

---

## 2. Experimental Context and Data

The project uses bearing life-test measurements from the **Paderborn University Bearing DataCenter**. The preprocessing code supports experiments `B01` through `B17` and accounts for the source sampling-rate groups: `B01`–`B09` are treated as 128 kHz measurements, and `B10`–`B17` as 64 kHz measurements.

The raw dataset is not included in this repository. Place each experiment under `data/raw_data/` with this structure:

```text
data/raw_data/B02/
├── vibrationData/
│   └── data_B02_MXXXX.mat
├── B02_operatingConditions.csv
└── B02_meanTemperatures.csv
```

The MATLAB measurements must include `measTime`, `accHorizRear_A`, and `accHorizFrontal_C`. The operating-condition CSV must include `meanAbs_speed / rpm`, `meanAbs_statLoad / N`, and `peak_dynLoad / N`; both CSV files must provide timestamps in a `Time` or `dateTime` column.

The `clean14` cohort used by default for training contains `B02`, `B03`, `B04`, `B06`–`B15`, and `B17`. The preprocessing script also supports the full `benchmark17` cohort.

---

## 3. Signal Processing Strategy

The project has two distinct preprocessing paths: a tensor pipeline for model training and a 4 kHz Parquet pipeline for exploratory data analysis (EDA).

### 3.1 Training Tensor Pipeline

For each experiment, the pipeline aligns operating and temperature metadata, reads vibration measurements in numeric file order, and skips measurements before the experiment's configured transition point. It converts the two horizontal acceleration channels to `g`, applies FIR decimation to the requested sample rate, and removes each signal's constant offset.

The default settings use 4,096-sample windows and sequences of 10 consecutive windows. Each sequence is paired with three operating features: mean absolute speed, mean absolute static load, and peak dynamic load. The target `y_pct` is the normalized remaining cycle fraction after the transition point; `y_s` also stores the time-based RUL target in seconds.

The compressed NPZ shards are written to `data/processed_data/BXX_tensors/`. The default sample rate is 32 kHz; window size, sequence length, and sample rate can be changed through command-line options.

### 3.2 Exploratory Parquet Pipeline

The separate EDA helper, `preprocess_to_parquet_eda`, creates a 4 kHz Parquet dataset with vibration channels, operating conditions, temperature measurements, and time-based RUL. It writes `eda_BXX_4kHz.parquet` under `data/processed_data/eda_parquet/`. This helper is available in [`src/preprocessing.py`](src/preprocessing.py); it is not one of the preprocessing script's command-line modes.

Run it for one experiment from the repository root, replacing `B02` as needed:

```bash
python -c "from src.preprocessing import preprocess_to_parquet_eda; preprocess_to_parquet_eda('B02')"
```

---

## 4. Model and Implementation

[`src/model.py`](src/model.py) defines an out-of-core dataset that loads one NPZ shard at a time and a CNN–LSTM fusion model. The CNN extracts features from each vibration window, the LSTM models the sequence of windows, and a regression head combines the final sequence representation with the normalized operating features to predict the remaining cycle fraction.

### 4.1 Preprocessing and Training

Run commands from the repository root:

```bash
# Generate tensors for the default 14-experiment training cohort
python src/preprocessing.py --split clean14

# Alternatively, preprocess all 17 supported experiments
python src/preprocessing.py --split benchmark17

# Train and validate on every bearing in the default training cohort
python src/train.py --val ALL --seeds 1
```

Use `--exp B02,B03` to preprocess selected experiments instead of a split. The preprocessing defaults are a 4,096-sample window, a sequence length of 10, and a 32 kHz sample rate. The training script supports validation on one or more configured bearings with `--val B02` or `--val B02,B03`, and accepts comma-separated random seeds with `--seeds 1,2`. Training runs on CUDA when available and otherwise uses the CPU.

The training program performs leave-one-bearing-out validation for the selected validation bearings. It calculates operating-feature normalization statistics from the training bearings for each fold and saves checkpoints, epoch history, validation predictions, and summary metrics under `results/`.

### 4.2 Exploratory Notebooks

* [`notebooks/time_n_freq_domain_graphs.ipynb`](notebooks/time_n_freq_domain_graphs.ipynb) compares vibration signals in the time and frequency domains using raw MATLAB measurements.
* [`notebooks/results_eda.ipynb`](notebooks/results_eda.ipynb) plots operating, temperature, vibration, and measurement-interval trends using the 4 kHz EDA Parquet datasets.

### 4.3 Repository Layout

```text
.
├── data/
│   ├── raw_data/                 # User-provided Paderborn measurements (not tracked)
│   └── processed_data/           # Generated NPZ tensors and EDA Parquet files (not tracked)
├── notebooks/                    # Signal visualization and EDA
├── results/                      # Checkpoints, logs, predictions, and metrics (not tracked)
├── src/
│   ├── model.py                  # Dataset and CNN–LSTM fusion model
│   ├── preprocessing.py          # Tensor and EDA preprocessing pipelines
│   └── train.py                  # Training and leave-one-bearing-out evaluation
├── .gitignore
├── LICENSE                       # MIT License
├── project_setup.py              # Creates the project directory structure
├── requirements.txt
└── README.md
```

Raw and processed datasets and model results are excluded from version control by `.gitignore`; only `.gitkeep` files preserve the empty data and results directories in a fresh checkout.
