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
    "gru_enhanced_total_cash_"
    "w60_h7_hs*_seed42.json"
)


for json_file in METRIC_DIR.glob(pattern):

    with open(json_file, "r") as f:
        data = json.load(f)

    if "config" not in data:
        continue

    config = data["config"]
    val = data["validation"]

    rows.append({
        "hidden_size": config["hidden_size"],
        "layers": config["num_layers"],
        "dropout": config["dropout"],

        "MAE": val["MAE"],
        "RMSE": val["RMSE"],
        "MAPE": val["MAPE"],
        "R2": val["R2"],

        "file": json_file.name,
    })


if len(rows) == 0:

    raise RuntimeError(
        "No GRU W60 H7 Total_Cash tuning files found."
    )


results = pd.DataFrame(rows)

results = results.sort_values(
    by=[
        "MAE",
        "RMSE",
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


print("\n" + "=" * 100)

print(
    "GRU W60 H7 TOTAL_CASH ARCHITECTURE TUNING"
)

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
    METRIC_DIR
    / "gru_w60_h7_total_cash_tuning_summary.csv"
)


results.to_csv(
    output_path,
    index=False
)


print("\nSaved:")
print(output_path)