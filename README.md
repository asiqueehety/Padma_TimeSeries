# Padma Bridge Time Series Forecasting

Daily traffic volume and toll revenue forecasting for the Padma Bridge using LSTM and GRU sequence models trained on strict raw historical observations. Two prediction targets are supported: total vehicle count (`Total_Traffic`) and total collected revenue (`Total_Cash`), at two forecast horizons: 1 day and 7 days ahead.

---

## Project Structure

```
Padma_TimeSeries/
|
|-- data/
|   |-- raw/
|       |-- padma_toll_report_with_holidays_weather.csv   # Primary dataset (required)
|       |-- jamuna_toll_report.csv                        # Reference dataset
|
|-- src/
|   |-- data_utils.py           # CSV loading and date parsing
|   |-- enhanced_features.py    # prepare_timeseries_dataframe() entry point
|   |-- sequence_data.py        # Sliding window sequence construction, raw feature list
|   |-- models.py               # SequenceRegressor (LSTM / GRU) definition
|   |-- train_utils.py          # EarlyStopping, loss evaluation, metrics
|
|-- outputs/
|   |-- checkpoints/            # Saved model weights and scalers (12 final dirs)
|   |-- predictions/            # Per-seed and ensemble prediction CSVs
|   |-- metrics/                # Final JSON and summary CSV
|   |-- figures/                # Final ensemble forecast plots
|   |-- live_predictions/       # Saved outputs from predict_single.py
|
|-- experiments/                # Model-selection scripts (not needed to run)
|   |-- run_raw_window_sweep.py
|   |-- summarize_raw_window_sweep.py
|   |-- run_raw_arch_tuning.py
|   |-- summarize_raw_arch_tuning.py
|   |-- summarize_raw_multiseed.py
|   |-- summarize_raw_final_candidates.py
|   |-- run_diagnostics.py
|   |-- src/diagnostics.py
|
|-- train_model.py              # Single training run (full CLI)
|-- run_raw_final_all.py        # Trains all 4 final configs × 3 seeds (12 runs)
|-- ensemble_raw_final.py       # Ensemble averaging of per-seed predictions
|-- predict_single.py           # Live inference on a specific date
|-- requirements.txt
|-- README.md
|-- .gitignore
```

Outputs are written to `outputs/` only. All output directories are excluded from version control.

---

## Final Model Configurations

Four configurations were selected after systematic window and architecture sweeps. Each is trained with seeds 1, 7, and 42; predictions are averaged across the three seeds to form the final ensemble.

| Task | Model | Window | Hidden | Dropout |
|---|---|---|---|---|
| Traffic H1 (1-day traffic) | GRU | 14 | 96 | 0.10 |
| Cash H1 (1-day revenue) | GRU | 7 | 64 | 0.20 |
| Traffic H7 (7-day traffic) | LSTM | 7 | 96 | 0.30 |
| Cash H7 (7-day revenue) | LSTM | 7 | 96 | 0.30 |

All models use `--input_type raw` and `--num_layers 1`.

---

## Requirements

- Python 3.10 or newer
- PyTorch 2.2 or newer (CPU is sufficient; CUDA is used automatically if available)

All Python dependencies:

```
numpy>=1.26
pandas>=2.2
scikit-learn>=1.4
scipy>=1.11
statsmodels>=0.14
matplotlib>=3.8
torch>=2.2
joblib>=1.3
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/asiqueehety/Padma_TimeSeries.git
cd Padma_TimeSeries
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the data files

Create the `data/raw/` directory and place the CSV files inside it:

```
data/raw/padma_toll_report_with_holidays_weather.csv
data/raw/jamuna_toll_report.csv
```

The Padma file must contain at minimum the columns: `Date`, `Traffic_Mawa`, `Traffic_Jajira`, `Cash_Mawa`, `Cash_Jajira`, `Total_Traffic`, `Total_Cash`, `temp_mean_c`, `temp_max_c`, `temp_min_c`, `rainfall_mm`, `humidity_pct`, `wind_speed_kmh`. Dates are formatted as `DD-MM-YY`.

---

## Training

### Train all final models (batch)

This trains the four finalized configurations each across three seeds (1, 7, 42) — 12 runs total:

```bash
python run_raw_final_all.py
```

Checkpoints are written to `outputs/checkpoints/`. Each checkpoint directory name encodes all hyperparameters and the seed.

### Single training run

`train_model.py` is the main entry point. Every hyperparameter is passed via command-line flags.

Example — reproducing the Traffic H1 final model:

```bash
python train_model.py \
    --model GRU \
    --input_type raw \
    --target Total_Traffic \
    --window 14 \
    --horizon 1 \
    --hidden_size 96 \
    --num_layers 1 \
    --dropout 0.10 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --patience 15 \
    --seed 42
```

**Flag reference:**

| Flag | Choices / Type | Default | Description |
|---|---|---|---|
| `--model` | `LSTM`, `GRU` | `LSTM` | Recurrent cell type |
| `--input_type` | `raw` | `raw` | Feature set. `raw` uses only historical observed variables |
| `--target` | `Total_Traffic`, `Total_Cash` | `Total_Traffic` | Prediction target column |
| `--window` | int | `30` | Look-back window in days |
| `--horizon` | int | `1` | Forecast horizon in days |
| `--hidden_size` | int | `64` | Hidden units per RNN layer |
| `--num_layers` | int | `1` | Number of stacked RNN layers |
| `--dropout` | float | `0.20` | Dropout rate |
| `--learning_rate` | float | `1e-3` | Adam learning rate |
| `--weight_decay` | float | `1e-4` | Adam weight decay |
| `--batch_size` | int | `32` | Mini-batch size |
| `--epochs` | int | `200` | Maximum training epochs |
| `--patience` | int | `15` | Early stopping patience (validation loss) |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--skip_test` | flag | off | Skip test-set evaluation (used during sweeps) |

A checkpoint directory is created at `outputs/checkpoints/<experiment_name>/` and contains:

- `model.pt` — model weights and metadata
- `sequence_scaler.joblib` — fitted StandardScaler for sequence features
- `target_scaler.joblib` — fitted StandardScaler for the target

Figures are written to `outputs/figures/` and metrics to `outputs/metrics/`.

---

## Ensemble

After multi-seed training is complete, run the ensemble script to average predictions across seeds and produce final metrics and plots:

```bash
python ensemble_raw_final.py
```

This reads per-seed prediction CSVs from `outputs/predictions/`, averages them across the three seeds, and writes ensemble metrics to `outputs/metrics/` and ensemble plots to `outputs/figures/`.

---

## Inference on a Specific Date

`predict_single.py` loads the saved checkpoints for all three seeds and runs an ensemble inference for a specific target date.

```bash
python predict_single.py --target Total_Traffic --horizon 1 --date 2026-01-15
```

```bash
python predict_single.py --target Total_Cash --horizon 7 --date 2026-01-15
```

The script prints the per-seed predictions and the ensemble average. Checkpoints must exist under `outputs/checkpoints/` before running. If the actual value is available in the dataset, the script also prints absolute and percentage error.

The prediction is saved to `outputs/live_predictions/`.

---

## Data Split

The chronological split is fixed in `train_model.py`:

| Split | Date Range |
|---|---|
| Train | start of data to 2025-04-22 |
| Validation | 2025-04-23 to 2025-12-22 |
| Test | 2025-12-23 onwards |

All scalers are fit exclusively on training data. Validation and test sets are only transformed, never used for fitting.

---

## Feature Engineering

**Raw feature set** (`--input_type raw`): The eight historical observed variables used as sequence input:

```
Total_Traffic
Total_Cash
temp_mean_c
temp_max_c
temp_min_c
rainfall_mm
humidity_pct
wind_speed_kmh
```

No calendar features, no lag features, no rolling statistics, and no future-known features are used. The sequence input is strictly what was observed on each day in the look-back window.

---

## Model Architecture

`SequenceRegressor` in `src/models.py` is a shared architecture for both LSTM and GRU:

1. A multi-layer RNN (`nn.LSTM` or `nn.GRU`) processes the look-back window. The representation from the final time step is used.
2. LayerNorm and dropout are applied to the sequence representation.
3. A two-layer MLP head with GELU activation produces a scalar prediction.

The model outputs a single normalized value. The target scaler is used to invert normalization when reporting metrics.

---

## Reproducibility

Set `--seed` to a fixed integer. The seed controls Python's `random` module, NumPy, and PyTorch including CUDA seeds. CuDNN deterministic mode is enabled automatically. To reproduce the published results exactly, use seeds 1, 7, and 42 with the configurations in `run_raw_final_all.py`.

---

## Notes

- Strict time-series discipline is enforced throughout: no shuffling before splitting, scalers fitted on training data only.
- Sequence windows containing NaN in any input feature column are dropped during dataset construction and do not appear in any split.
- CUDA is detected and used automatically. No code changes are needed to switch between CPU and GPU.
- Checkpoint directory names encode all hyperparameters so that concurrent runs with different settings never overwrite each other.
- The `experiments/` folder contains all window sweep and architecture tuning scripts used to select the final configurations. These are preserved for scientific reproducibility but are not needed to run the final project.
