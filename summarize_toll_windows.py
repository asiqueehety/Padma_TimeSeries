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


for model in ["lstm", "gru"]:

    pattern = (
        f"{model}_enhanced_total_cash_"
        f"w*_h1_hs64_nl1_do0p2_*_seed42.json"
    )

    for json_file in METRIC_DIR.glob(pattern):

        with open(
            json_file,
            "r"
        ) as f:

            data = json.load(f)

        if "config" not in data:
            continue

        config = data["config"]
        validation = data["validation"]

        rows.append(
            {
                "model": config["model"],
                "window": config["window"],
                "MAE": validation["MAE"],
                "RMSE": validation["RMSE"],
                "MAPE": validation["MAPE"],
                "R2": validation["R2"],
            }
        )


if len(rows) == 0:

    raise RuntimeError(
        "No Total_Cash window experiment JSON files found."
    )


results = pd.DataFrame(rows)

results = results.sort_values(
    by=[
        "model",
        "MAE"
    ],
    ascending=[
        True,
        True
    ]
)


pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    160
)


print("\n")
print("=" * 90)
print("TOTAL_CASH WINDOW VALIDATION RESULTS")
print("=" * 90)

print(
    results.to_string(
        index=False
    )
)


output_path = (
    METRIC_DIR
    / "total_cash_window_summary.csv"
)

results.to_csv(
    output_path,
    index=False
)


print(
    "\nSaved:"
)

print(
    output_path
)