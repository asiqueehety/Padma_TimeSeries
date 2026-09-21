from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

PREDICTION_DIR = (
    ROOT
    / "outputs"
    / "predictions"
)


# ============================================================
# FIND THE THREE FINAL H7 TRAFFIC PREDICTIONS
# ============================================================

pattern = (
    "gru_enhanced_total_traffic_"
    "w90_h7_hs64_nl1_do0p1_*_seed*.csv"
)

files = sorted(
    PREDICTION_DIR.glob(pattern)
)


print("\nFound prediction files:")

for file in files:
    print(file.name)


if len(files) != 3:

    raise RuntimeError(
        f"\nExpected exactly 3 prediction files "
        f"for seeds 1, 7 and 42, "
        f"but found {len(files)}."
    )


# ============================================================
# LOAD FIRST FILE
# ============================================================

base = pd.read_csv(
    files[0]
)

base["target_date"] = pd.to_datetime(
    base["target_date"]
)


ensemble = base[
    [
        "target_date",
        "actual",
    ]
].copy()


prediction_columns = []


# ============================================================
# LOAD ALL THREE SEEDS
# ============================================================

for i, file in enumerate(
    files,
    start=1
):

    current = pd.read_csv(
        file
    )

    current["target_date"] = pd.to_datetime(
        current["target_date"]
    )

    # Make sure all three runs predict the same dates.
    if not np.array_equal(
        current["target_date"].values,
        ensemble["target_date"].values,
    ):

        raise RuntimeError(
            f"Target dates do not match in {file.name}"
        )

    # Make sure actual values also match.
    if not np.allclose(
        current["actual"].values,
        ensemble["actual"].values,
    ):

        raise RuntimeError(
            f"Actual values do not match in {file.name}"
        )

    column_name = (
        f"prediction_seed_{i}"
    )

    ensemble[column_name] = (
        current["prediction"].values
    )

    prediction_columns.append(
        column_name
    )


# ============================================================
# THREE-SEED ENSEMBLE
# ============================================================

ensemble[
    "ensemble_prediction"
] = (
    ensemble[
        prediction_columns
    ]
    .mean(axis=1)
)


# ============================================================
# METRICS
# ============================================================

y_true = (
    ensemble[
        "actual"
    ].values
)

y_pred = (
    ensemble[
        "ensemble_prediction"
    ].values
)


mae = mean_absolute_error(
    y_true,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred
    )
)

mape = (
    mean_absolute_percentage_error(
        y_true,
        y_pred
    )
    * 100
)

r2 = r2_score(
    y_true,
    y_pred
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n"
    + "=" * 72
)

print(
    "GRU W90 H7 TRAFFIC - 3 SEED ENSEMBLE"
)

print(
    "=" * 72
)

print(
    f"MAE :  {mae:.4f}"
)

print(
    f"RMSE:  {rmse:.4f}"
)

print(
    f"MAPE:  {mape:.4f}%"
)

print(
    f"R2  :  {r2:.4f}"
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    PREDICTION_DIR
    / "gru_enhanced_total_traffic_w90_h7_seed_ensemble.csv"
)


ensemble.to_csv(
    output_path,
    index=False
)


print(
    "\nSaved ensemble file:"
)

print(
    output_path
)