from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)


ROOT = Path(__file__).resolve().parent

PREDICTION_DIR = (
    ROOT
    / "outputs"
    / "predictions"
)


pattern = (
    "lstm_enhanced_total_cash_"
    "w14_h1_hs96_nl1_do0p1_*_seed*.csv"
)

files = sorted(
    PREDICTION_DIR.glob(pattern)
)


print("Found prediction files:")

for file in files:
    print(file.name)


if len(files) != 3:
    raise RuntimeError(
        f"Expected exactly 3 prediction files, "
        f"but found {len(files)}."
    )


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

    if not np.array_equal(
        current["target_date"].values,
        ensemble["target_date"].values,
    ):
        raise RuntimeError(
            f"Target dates do not match in {file.name}"
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


ensemble[
    "ensemble_prediction"
] = (

    ensemble[
        prediction_columns
    ]
    .mean(axis=1)
)


y_true = ensemble[
    "actual"
].values

y_pred = ensemble[
    "ensemble_prediction"
].values


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


print("\n" + "=" * 72)
print("3-SEED LSTM TOLL ENSEMBLE TEST RESULTS")
print("=" * 72)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"R2: {r2:.4f}")


output_path = (
    PREDICTION_DIR
    / "lstm_enhanced_total_cash_w14_h1_seed_ensemble.csv"
)

ensemble.to_csv(
    output_path,
    index=False
)

print("\nSaved:")
print(output_path)