# Padma Bridge Time Series Forecasting

Daily traffic volume and toll revenue forecasting for the Padma Bridge using LSTM and GRU sequence models. The models are trained on Padma toll report data augmented with Jamuna Bridge data as a correlated cross-bridge signal. Two prediction targets are supported: total vehicle count (Total_Traffic) and total collected revenue (Total_Cash), at two forecast horizons: 1 day and 7 days ahead.

---

## Project Structure

```
Padma_TimeSeries/
|
|-- data/
|   |-- raw/
|       |-- padma_toll_report_with_holidays_weather.csv   # Primary dataset (required)
|       |-- jamuna_toll_report.csv                        # Secondary dataset (required)
|
|-- src/
|   |-- data_utils.py           # CSV loading, date parsing, calendar and Eid features
|   |-- enhanced_features.py    # Cross-bridge features, lagged targets, rolling statistics
|   |-- sequence_data.py        # Sliding window sequence construction
|   |-- models.py               # SequenceRegressor (LSTM / GRU) definition
|   |-- train_utils.py          # EarlyStopping, loss evaluation, metrics
|   |-- diagnostics.py          # Optional data inspection utilities
|
|-- train_model.py              # Single training run (full CLI)
|-- predict_single.py           # Inference on a saved checkpoint for a specific date
|-- run_raw_final_all.py        # Batch runner: all 4 tasks x 3 seeds (raw features)
|-- run_raw_window_sweep.py     # Window-size sweep over all tasks
|-- run_raw_arch_tuning.py      # Architecture hyperparameter sweep
|-- summarize_raw_multiseed.py  # Aggregate and compare multi-seed results
|-- summarize_raw_final_candidates.py
|-- summarize_raw_arch_tuning.py
|-- summarize_raw_window_sweep.py
|-- ensemble_raw_final.py       # Ensemble averaging of multi-seed predictions
|-- prepare_demo_outputs.py     # Prepare presentation-ready plots and CSVs
|-- run_diagnostics.py          # Run data diagnostics
|-- requirements.txt
```

Outputs (checkpoints, figures, metrics, predictions) are written to `outputs/`, `enhanced_outputs/`, or `archive_outputs/` depending on the run configuration. All output directories are excluded from version control.

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

Create the `data/raw/` directory and place the two CSV files inside it:

```
data/raw/padma_toll_report_with_holidays_weather.csv
data/raw/jamuna_toll_report.csv
```

The Padma file must contain at minimum the columns: `Date`, `Traffic_Mawa`, `Traffic_Jajira`, `Cash_Mawa`, `Cash_Jajira`, `Total_Traffic`, `Total_Cash`, `holiday_name`, `days_to_nearest_eid`. Dates are formatted as `DD-MM-YY`.

The Jamuna file must contain: `Date`, `Traffic_East`, `Traffic_West`, `Cash_East`, `Cash_West`, `Total_Traffic`, `Total_Cash`. Dates are formatted as `DD/MM/YYYY`.

---

## Training

### Single training run

`train_model.py` is the main entry point. Every hyperparameter is passed via command-line flags.

```bash
python train_model.py \
    --model LSTM \
    --input_type enhanced \
    --target Total_Traffic \
    --window 30 \
    --horizon 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --dropout 0.20 \
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
| `--input_type` | `raw`, `enhanced` | `enhanced` | Feature set. `raw` uses only base Padma columns; `enhanced` adds Jamuna cross-bridge features, lagged targets, and rolling statistics |
| `--target` | `Total_Traffic`, `Total_Cash` | `Total_Traffic` | Prediction target column |
| `--window` | int | `30` | Look-back window in days |
| `--horizon` | int | `1` | Forecast horizon in days |
| `--hidden_size` | int | `64` | Hidden units per RNN layer |
| `--num_layers` | int | `1` | Number of stacked RNN layers |
| `--dropout` | float | `0.20` | Dropout rate (applied between layers and in the regression head) |
| `--learning_rate` | float | `1e-3` | Adam learning rate |
| `--weight_decay` | float | `1e-4` | Adam weight decay |
| `--batch_size` | int | `32` | Mini-batch size |
| `--epochs` | int | `200` | Maximum training epochs |
| `--patience` | int | `15` | Early stopping patience (validation loss) |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--skip_test` | flag | off | Skip test-set evaluation (useful during hyperparameter search to avoid test contamination) |

A checkpoint directory is created at `<outputs_root>/checkpoints/<experiment_name>/` and contains:
- `model.pt` — model weights
- `sequence_scaler.joblib` — fitted StandardScaler for sequence features
- `future_scaler.joblib` — fitted StandardScaler for future-known features
- `target_scaler.joblib` — fitted StandardScaler for the target

Figures are written to `<outputs_root>/figures/` and metrics to `<outputs_root>/metrics/`.

The outputs root is `enhanced_outputs/` when `--input_type enhanced` and `outputs/` when `--input_type raw`.

### Train all final models (batch)

This trains the four finalized configurations (Traffic H1, Cash H1, Traffic H7, Cash H7) each across three seeds (1, 7, 42) using raw features — 12 runs total:

```bash
python run_raw_final_all.py
```

To train with enhanced features instead, call `train_model.py` directly with `--input_type enhanced` for each configuration.

### Hyperparameter sweeps

Window sweep (varies look-back window for each task):

```bash
python run_raw_window_sweep.py
```

Architecture sweep (varies hidden size, number of layers, dropout):

```bash
python run_raw_arch_tuning.py
```

Both scripts pass `--skip_test` internally to avoid using the test set during search.

---

## Evaluating and Summarizing Results

After training runs complete, use the summary scripts to aggregate metrics across seeds and configurations.

Multi-seed summary (reads per-seed metric JSON files, computes mean and standard deviation across seeds):

```bash
python summarize_raw_multiseed.py
```

Candidate summary (compares window sweep results to identify the best window per task):

```bash
python summarize_raw_final_candidates.py
```

Architecture tuning summary:

```bash
python summarize_raw_arch_tuning.py
```

Window sweep summary:

```bash
python summarize_raw_window_sweep.py
```

Each script prints a formatted table to stdout. Redirect to a file to save results:

```bash
python summarize_raw_multiseed.py > results_summary.txt
```

---

## Ensemble

After multi-seed training is complete, run the ensemble script to average predictions across seeds and produce final metrics and plots:

```bash
python ensemble_raw_final.py
```

This reads per-seed prediction CSVs from `outputs/predictions/`, averages them, and writes ensemble metrics to `outputs/metrics/` and ensemble plots to `outputs/figures/`.

---

## Inference on a Specific Date

`predict_single.py` loads saved checkpoints and runs inference for a specific target date. It uses the finalized model configurations defined at the top of the script.

```bash
python predict_single.py --target Total_Traffic --horizon 1 --date 2026-01-15
```

```bash
python predict_single.py --target Total_Cash --horizon 7 --date 2026-01-15
```

The script prints the ensemble prediction averaged over the three seeds, plus a per-seed breakdown. Checkpoints are resolved automatically from the configured experiment names. The checkpoint files must exist under `enhanced_outputs/checkpoints/` before running.

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

**Raw feature set** (`--input_type raw`): The base Padma columns (Traffic_Mawa, Traffic_Jajira, Cash_Mawa, Cash_Jajira, Total_Traffic, Total_Cash) plus calendar features (day-of-week, month, day-of-year encoded as sine/cosine pairs), a linear time index, signed Eid relative day, and railway opening indicator.

**Enhanced feature set** (`--input_type enhanced`): All raw features plus Jamuna Bridge totals (cross-bridge traffic and revenue), lagged target values at lags 1, 2, 3, 7, 14, 21, and 28 days, and rolling means and standard deviations over 7- and 14-day windows. Missing historical values in sequence windows are filled causally (seasonal lag of 7 days then forward fill) to avoid any future leakage.

---

## Model Architecture

`SequenceRegressor` in `src/models.py` is a shared architecture for both LSTM and GRU:

1. A multi-layer RNN (`nn.LSTM` or `nn.GRU`) processes the look-back window. The representation from the final time step is used.
2. LayerNorm and dropout are applied to the sequence representation.
3. An optional small MLP branch processes future-known features (calendar features for the target date). Its output is concatenated to the sequence representation when present.
4. A two-layer MLP head with GELU activation produces a scalar prediction.

The model outputs a single normalized value. The target scaler is used to invert normalization when reporting metrics.

---

## Extending the Project

**Add a new prediction target**: Add the column name to the `--target` choices list in `train_model.py` and ensure the column is present after `prepare_timeseries_dataframe()` runs.

**Add new input features**: Modify `src/enhanced_features.py`. The function `prepare_timeseries_dataframe()` is the single entry point for all feature construction. Any new column added there becomes part of the sequence feature set automatically.

**Change the train/validation/test split dates**: Modify the date constants in the `split_indices` function in `train_model.py`.

**Add a new model type**: Add the type name to the `model_type` choices in `train_model.py` and add the corresponding `nn.Module` branch inside `SequenceRegressor.__init__` in `src/models.py`.

**Multi-step direct output**: Currently each model predicts one scalar per forward pass regardless of horizon (the horizon affects which target date is constructed, not the output dimension). To switch to a direct multi-step output head, modify the head output dimension in `src/models.py` and the target construction in `src/sequence_data.py`.

---

## Reproducibility

Set `--seed` to a fixed integer. The seed controls Python's `random` module, NumPy, and PyTorch including CUDA seeds. CuDNN deterministic mode is enabled automatically. To reproduce the published results exactly, use seeds 1, 7, and 42 with the configurations in `run_raw_final_all.py`.

---

## Notes

- Strict time-series discipline is enforced throughout: no shuffling before splitting, scalers fitted on training data only, causal feature filling with no look-ahead.
- Sequence windows containing NaN in any input feature column are dropped during dataset construction and do not appear in any split.
- CUDA is detected and used automatically. No code changes are needed to switch between CPU and GPU.
- Checkpoint directory names encode all hyperparameters so that concurrent runs with different settings never overwrite each other.
