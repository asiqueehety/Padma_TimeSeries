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


for json_file in METRIC_DIR.glob(
    "lstm_enhanced_total_traffic_w30_h1_hs*_seed42.json"
):

    with open(
        json_file,
        "r"
    ) as f:

        data = json.load(f)


    # Only new-format JSON files
    if "config" not in data:
        continue


    config = data["config"]
    validation = data["validation"]


    rows.append({

        "file":
            json_file.name,

        "hidden_size":
            config["hidden_size"],

        "layers":
            config["num_layers"],

        "dropout":
            config["dropout"],

        "learning_rate":
            config["learning_rate"],

        "MAE":
            validation["MAE"],

        "RMSE":
            validation["RMSE"],

        "MAPE":
            validation["MAPE"],

        "R2":
            validation["R2"],
    })


if len(rows) == 0:

    raise RuntimeError(
        "No tuning JSON files found."
    )


results = pd.DataFrame(rows)


# Primary ranking = MAE.
# RMSE breaks close ties.
results = results.sort_values(
    by=[
        "MAE",
        "RMSE"
    ],
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
print("LSTM W30 VALIDATION TUNING RESULTS")
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
    ].to_string(
        index=False
    )
)


output_path = (
    ROOT
    / "outputs"
    / "metrics"
    / "lstm_w30_tuning_summary.csv"
)

results.to_csv(
    output_path,
    index=False
)


print(
    "\nSaved summary to:"
)

print(
    output_path
)