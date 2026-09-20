from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parent

METRIC_DIR = (
    ROOT
    / "outputs"
    / "metrics"
)

rows = []

pattern = (
    "lstm_enhanced_total_cash_"
    "w14_h1_hs*_seed42.json"
)

for json_file in METRIC_DIR.glob(pattern):

    with open(json_file, "r") as f:
        data = json.load(f)

    if "config" not in data:
        continue

    config = data["config"]
    validation = data["validation"]

    rows.append({
        "file": json_file.name,
        "hidden_size": config["hidden_size"],
        "layers": config["num_layers"],
        "dropout": config["dropout"],
        "learning_rate": config["learning_rate"],
        "MAE": validation["MAE"],
        "RMSE": validation["RMSE"],
        "MAPE": validation["MAPE"],
        "R2": validation["R2"],
    })


if len(rows) == 0:
    raise RuntimeError(
        "No LSTM Total_Cash W14 tuning files found."
    )


results = pd.DataFrame(rows)

results = results.sort_values(
    by=["MAE", "RMSE"],
    ascending=True
)

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)

print("\n")
print("=" * 100)
print("LSTM W14 TOTAL_CASH TUNING RESULTS")
print("=" * 100)

print(
    results[
        [
            "hidden_size",
            "layers",
            "dropout",
            "MAE",
            "RMSE",
            "MAPE",
            "R2",
        ]
    ].to_string(index=False)
)


output_path = (
    METRIC_DIR
    / "lstm_w14_total_cash_tuning_summary.csv"
)

results.to_csv(
    output_path,
    index=False
)

print("\nSaved summary:")
print(output_path)