from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

SOURCE_OUTPUTS = ROOT / "outputs"

CLEAN_OUTPUTS = ROOT / "outputs_demo_clean"

SOURCE_CHECKPOINTS = SOURCE_OUTPUTS / "checkpoints"
SOURCE_PREDICTIONS = SOURCE_OUTPUTS / "predictions"

CLEAN_CHECKPOINTS = CLEAN_OUTPUTS / "checkpoints"
CLEAN_PREDICTIONS = CLEAN_OUTPUTS / "predictions"
CLEAN_METRICS = CLEAN_OUTPUTS / "metrics"
CLEAN_FIGURES = CLEAN_OUTPUTS / "figures"
CLEAN_LIVE = CLEAN_OUTPUTS / "live_predictions"


# ============================================================
# FINAL MODELS ONLY
# ============================================================

FINAL_MODELS = {

    "traffic_h1": {
        "experiment_base":
            "lstm_enhanced_total_traffic_"
            "w30_h1_hs64_nl1_do0p2_"
            "lr1em03_wd1em04",

        "ensemble_name":
            "lstm_enhanced_total_traffic_"
            "w30_h1_seed_ensemble.csv",

        "title":
            "Traffic - 1 Day Ahead",

        "ylabel":
            "Total Traffic",
    },

    "cash_h1": {
        "experiment_base":
            "lstm_enhanced_total_cash_"
            "w14_h1_hs96_nl1_do0p1_"
            "lr1em03_wd1em04",

        "ensemble_name":
            "lstm_enhanced_total_cash_"
            "w14_h1_seed_ensemble.csv",

        "title":
            "Toll - 1 Day Ahead",

        "ylabel":
            "Total Cash (BDT)",
    },

    "traffic_h7": {
        "experiment_base":
            "gru_enhanced_total_traffic_"
            "w90_h7_hs64_nl1_do0p1_"
            "lr1em03_wd1em04",

        "ensemble_name":
            "gru_enhanced_total_traffic_"
            "w90_h7_seed_ensemble.csv",

        "title":
            "Traffic - 7 Days Ahead",

        "ylabel":
            "Total Traffic",
    },

    "cash_h7": {
        "experiment_base":
            "gru_enhanced_total_cash_"
            "w60_h7_hs64_nl1_do0p2_"
            "lr1em03_wd1em04",

        "ensemble_name":
            "gru_enhanced_total_cash_"
            "w60_h7_seed_ensemble.csv",

        "title":
            "Toll - 7 Days Ahead",

        "ylabel":
            "Total Cash (BDT)",
    },
}


SEEDS = [1, 7, 42]


# ============================================================
# CREATE CLEAN FOLDERS
# ============================================================

for folder in [
    CLEAN_CHECKPOINTS,
    CLEAN_PREDICTIONS,
    CLEAN_METRICS,
    CLEAN_FIGURES,
    CLEAN_LIVE,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# PROCESS EACH FINAL MODEL
# ============================================================

for short_name, config in FINAL_MODELS.items():

    print("\n" + "=" * 80)
    print(short_name.upper())
    print("=" * 80)

    experiment_base = config["experiment_base"]

    # --------------------------------------------------------
    # COPY THREE FINAL CHECKPOINTS
    # --------------------------------------------------------

    for seed in SEEDS:

        experiment_name = (
            f"{experiment_base}_seed{seed}"
        )

        source_dir = (
            SOURCE_CHECKPOINTS
            / experiment_name
        )

        destination_dir = (
            CLEAN_CHECKPOINTS
            / experiment_name
        )

        if not source_dir.exists():

            raise FileNotFoundError(
                f"\nMissing checkpoint directory:\n"
                f"{source_dir}"
            )

        required_files = [
            "model.pt",
            "sequence_scaler.joblib",
            "future_scaler.joblib",
            "target_scaler.joblib",
        ]

        for required in required_files:

            file_path = (
                source_dir
                / required
            )

            if not file_path.exists():

                raise FileNotFoundError(
                    f"\nMissing required file:\n"
                    f"{file_path}"
                )

        if destination_dir.exists():
            shutil.rmtree(
                destination_dir
            )

        shutil.copytree(
            source_dir,
            destination_dir,
        )

        print(
            f"Copied checkpoint: "
            f"{experiment_name}"
        )

    # --------------------------------------------------------
    # BUILD / REBUILD ENSEMBLE FROM THREE SEED CSVs
    # --------------------------------------------------------

    seed_files = []

    for seed in SEEDS:

        prediction_name = (
            f"{experiment_base}_seed{seed}.csv"
        )

        prediction_path = (
            SOURCE_PREDICTIONS
            / prediction_name
        )

        if not prediction_path.exists():

            raise FileNotFoundError(
                f"\nMissing final test prediction:\n"
                f"{prediction_path}\n\n"
                f"The corresponding final model may need "
                f"to be run once without --skip_test."
            )

        seed_files.append(
            prediction_path
        )

    first = pd.read_csv(
        seed_files[0]
    )

    first["target_date"] = pd.to_datetime(
        first["target_date"]
    )

    ensemble = first[
        [
            "target_date",
            "actual",
        ]
    ].copy()

    for seed, file_path in zip(
        SEEDS,
        seed_files
    ):

        current = pd.read_csv(
            file_path
        )

        current["target_date"] = pd.to_datetime(
            current["target_date"]
        )

        if not np.array_equal(
            ensemble["target_date"].values,
            current["target_date"].values,
        ):
            raise RuntimeError(
                f"Target-date mismatch in "
                f"{file_path.name}"
            )

        if not np.allclose(
            ensemble["actual"].values,
            current["actual"].values,
        ):
            raise RuntimeError(
                f"Actual-value mismatch in "
                f"{file_path.name}"
            )

        ensemble[
            f"prediction_seed_{seed}"
        ] = (
            current["prediction"].values
        )

    ensemble[
        "ensemble_prediction"
    ] = ensemble[
        [
            "prediction_seed_1",
            "prediction_seed_7",
            "prediction_seed_42",
        ]
    ].mean(
        axis=1
    )

    ensemble_path = (
        CLEAN_PREDICTIONS
        / config["ensemble_name"]
    )

    ensemble.to_csv(
        ensemble_path,
        index=False,
    )

    print(
        f"Created ensemble: "
        f"{ensemble_path.name}"
    )

    # --------------------------------------------------------
    # FINAL METRICS
    # --------------------------------------------------------

    y_true = (
        ensemble["actual"]
        .to_numpy()
    )

    y_pred = (
        ensemble[
            "ensemble_prediction"
        ]
        .to_numpy()
    )

    metrics = {
        "MAE": float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        ),

        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            )
        ),

        "MAPE": float(
            mean_absolute_percentage_error(
                y_true,
                y_pred
            )
            * 100
        ),

        "R2": float(
            r2_score(
                y_true,
                y_pred
            )
        ),
    }

    metric_path = (
        CLEAN_METRICS
        / f"{short_name}_final.json"
    )

    with open(
        metric_path,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )

    print("\nFinal ensemble metrics:")

    for key, value in metrics.items():

        print(
            f"{key}: {value:.4f}"
        )

    # --------------------------------------------------------
    # FINAL ENSEMBLE GRAPH
    # --------------------------------------------------------

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        ensemble["target_date"],
        ensemble["actual"],
        label="Actual",
    )

    plt.plot(
        ensemble["target_date"],
        ensemble[
            "ensemble_prediction"
        ],
        label="Predicted",
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        config["ylabel"]
    )

    plt.title(
        config["title"]
        + " - Final 3-Seed Ensemble"
    )

    plt.legend()

    plt.tight_layout()

    figure_path = (
        CLEAN_FIGURES
        / f"{short_name}_final.png"
    )

    plt.savefig(
        figure_path,
        dpi=180,
    )

    plt.close()

    print(
        f"Created figure: "
        f"{figure_path.name}"
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("CLEAN DEMO OUTPUTS CREATED")
print("=" * 80)

print(
    f"\nLocation:\n{CLEAN_OUTPUTS}"
)

print(
    "\nDo NOT delete the original outputs folder "
    "until you verify this clean folder."
)