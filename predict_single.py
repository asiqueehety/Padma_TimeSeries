import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


# ============================================================
# PROJECT IMPORTS
# ============================================================

ROOT = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(ROOT)
)

sys.path.insert(
    0,
    str(ROOT / "src")
)


from src.enhanced_features import (
    prepare_timeseries_dataframe,
)

from src.sequence_data import (
    get_sequence_features,
    FUTURE_KNOWN_FEATURES,
)

from src.models import (
    SequenceRegressor,
)


# ============================================================
# FINAL MODEL CONFIGURATIONS
# ============================================================

FINAL_CONFIGS = {

    # --------------------------------------------------------
    # 1-DAY TRAFFIC
    # --------------------------------------------------------
    ("Total_Traffic", 1): {
        "model": "LSTM",
        "window": 30,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.20,
    },

    # --------------------------------------------------------
    # 1-DAY TOLL
    # --------------------------------------------------------
    ("Total_Cash", 1): {
        "model": "LSTM",
        "window": 14,
        "hidden_size": 96,
        "num_layers": 1,
        "dropout": 0.10,
    },

    # --------------------------------------------------------
    # 7-DAY TRAFFIC
    # --------------------------------------------------------
    ("Total_Traffic", 7): {
        "model": "GRU",
        "window": 90,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.10,
    },

    # --------------------------------------------------------
    # 7-DAY TOLL
    # --------------------------------------------------------
    ("Total_Cash", 7): {
        "model": "GRU",
        "window": 60,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.20,
    },
}


SEEDS = [
    1,
    7,
    42,
]


INPUT_TYPE = "enhanced"

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


# ============================================================
# BUILD EXPERIMENT NAME EXACTLY LIKE train_model.py
# ============================================================

def build_experiment_name(
    model,
    target,
    window,
    horizon,
    hidden_size,
    num_layers,
    dropout,
    seed,
):

    dropout_tag = (
        str(dropout)
        .replace(".", "p")
    )

    lr_tag = (
        f"{LEARNING_RATE:.0e}"
        .replace("-", "m")
    )

    wd_tag = (
        f"{WEIGHT_DECAY:.0e}"
        .replace("-", "m")
    )

    experiment_name = (

        f"{model.lower()}"

        f"_{INPUT_TYPE}"

        f"_{target.lower()}"

        f"_w{window}"

        f"_h{horizon}"

        f"_hs{hidden_size}"

        f"_nl{num_layers}"

        f"_do{dropout_tag}"

        f"_lr{lr_tag}"

        f"_wd{wd_tag}"

        f"_seed{seed}"
    )

    return experiment_name


# ============================================================
# LOAD TORCH CHECKPOINT
# ============================================================

def load_checkpoint(
    checkpoint_path,
    device,
):

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    return checkpoint


# ============================================================
# GET ONE EXACT TIME-SERIES INPUT
# ============================================================

def prepare_single_input(
    df,
    target_date,
    target,
    horizon,
    window,
):

    feature_columns = (
        get_sequence_features(
            INPUT_TYPE
        )
    )

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    target_date = pd.Timestamp(
        target_date
    )

    origin_date = (
        target_date
        -
        pd.Timedelta(
            days=horizon
        )
    )

    start_date = (
        origin_date
        -
        pd.Timedelta(
            days=window - 1
        )
    )

    # --------------------------------------------------------
    # Find target-date row
    # --------------------------------------------------------

    target_rows = df[
        df["Date"] == target_date
    ]

    if len(target_rows) == 0:

        raise ValueError(
            "\nForecast date is not present in the prepared "
            "daily dataframe.\n\n"
            "For this current demonstration script, choose "
            "a date already present in your dataset.\n"
        )

    target_row = (
        target_rows
        .iloc[0]
    )

    # --------------------------------------------------------
    # Select historical window
    # --------------------------------------------------------

    sequence_df = df[
        (
            df["Date"] >= start_date
        )
        &
        (
            df["Date"] <= origin_date
        )
    ].copy()

    sequence_df = (
        sequence_df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(sequence_df) != window:

        raise ValueError(
            f"\nExpected {window} historical days, "
            f"but found {len(sequence_df)}.\n"
            f"Required range: "
            f"{start_date.date()} -> "
            f"{origin_date.date()}"
        )

    # --------------------------------------------------------
    # Verify every day is consecutive
    # --------------------------------------------------------

    expected_dates = pd.date_range(
        start=start_date,
        end=origin_date,
        freq="D",
    )

    actual_dates = (
        sequence_df["Date"]
        .to_numpy(
            dtype="datetime64[ns]"
        )
    )

    if not np.array_equal(
        actual_dates,
        expected_dates.to_numpy(
            dtype="datetime64[ns]"
        ),
    ):

        raise ValueError(
            "Historical sequence is not "
            "strictly consecutive daily data."
        )

    # --------------------------------------------------------
    # Build sequence:
    #
    # 1 x window x features
    # --------------------------------------------------------

    sequence = (
        sequence_df[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    if np.isnan(
        sequence
    ).any():

        raise ValueError(
            "Historical input contains NaN values."
        )

    sequence = sequence[
        np.newaxis,
        :,
        :
    ]

    # --------------------------------------------------------
    # Target-date information known in advance
    # --------------------------------------------------------

    future_features = (
        target_row[
            FUTURE_KNOWN_FEATURES
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    if np.isnan(
        future_features
    ).any():

        raise ValueError(
            "Target-date future-known features "
            "contain NaN values."
        )

    future_features = (
        future_features[
            np.newaxis,
            :
        ]
    )

    # --------------------------------------------------------
    # Actual target is ONLY for display/comparison.
    #
    # It is NOT included in model input.
    # --------------------------------------------------------

    actual_value = (
        target_row[target]
    )

    if pd.isna(
        actual_value
    ):
        actual_value = None

    else:
        actual_value = float(
            actual_value
        )

    return (
        sequence,
        future_features,
        actual_value,
        start_date,
        origin_date,
        target_date,
        feature_columns,
    )


# ============================================================
# PREDICT USING ONE SAVED SEED
# ============================================================

def predict_one_seed(
    sequence,
    future_features,
    feature_columns,
    config,
    target,
    horizon,
    seed,
    device,
):

    experiment_name = (
        build_experiment_name(

            model=
                config["model"],

            target=
                target,

            window=
                config["window"],

            horizon=
                horizon,

            hidden_size=
                config["hidden_size"],

            num_layers=
                config["num_layers"],

            dropout=
                config["dropout"],

            seed=
                seed,
        )
    )

    checkpoint_dir = (
        ROOT
        / "outputs"
        / "checkpoints"
        / experiment_name
    )

    model_path = (
        checkpoint_dir
        / "model.pt"
    )

    sequence_scaler_path = (
        checkpoint_dir
        / "sequence_scaler.joblib"
    )

    future_scaler_path = (
        checkpoint_dir
        / "future_scaler.joblib"
    )

    target_scaler_path = (
        checkpoint_dir
        / "target_scaler.joblib"
    )

    required_files = [
        model_path,
        sequence_scaler_path,
        future_scaler_path,
        target_scaler_path,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                "\nRequired trained-model file "
                "was not found:\n"
                f"{file_path}\n\n"
                "Make sure the final training run "
                "for this seed exists."
            )

    # --------------------------------------------------------
    # Load saved training objects
    # --------------------------------------------------------

    checkpoint = (
        load_checkpoint(
            model_path,
            device,
        )
    )

    sequence_scaler = (
        joblib.load(
            sequence_scaler_path
        )
    )

    future_scaler = (
        joblib.load(
            future_scaler_path
        )
    )

    target_scaler = (
        joblib.load(
            target_scaler_path
        )
    )

    # --------------------------------------------------------
    # Make sure current features match training features
    # --------------------------------------------------------

    saved_features = (
        checkpoint[
            "feature_columns"
        ]
    )

    if list(
        saved_features
    ) != list(
        feature_columns
    ):

        raise RuntimeError(
            "Current feature order does not match "
            "the feature order used during training."
        )

    # --------------------------------------------------------
    # Scale exactly the same way as training
    # --------------------------------------------------------

    original_shape = (
        sequence.shape
    )

    sequence_flat = (
        sequence.reshape(
            -1,
            sequence.shape[-1]
        )
    )

    sequence_scaled = (
        sequence_scaler
        .transform(
            sequence_flat
        )
        .reshape(
            original_shape
        )
        .astype(
            np.float32
        )
    )

    future_scaled = (
        future_scaler
        .transform(
            future_features
        )
        .astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # Rebuild architecture
    # --------------------------------------------------------

    model = SequenceRegressor(

        input_size=
            checkpoint[
                "input_size"
            ],

        future_size=
            checkpoint[
                "future_size"
            ],

        model_type=
            checkpoint[
                "model_type"
            ],

        hidden_size=
            checkpoint[
                "hidden_size"
            ],

        num_layers=
            checkpoint[
                "num_layers"
            ],

        dropout=
            checkpoint[
                "dropout"
            ],
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    sequence_tensor = (
        torch.tensor(
            sequence_scaled,
            dtype=torch.float32,
        )
        .to(
            device
        )
    )

    future_tensor = (
        torch.tensor(
            future_scaled,
            dtype=torch.float32,
        )
        .to(
            device
        )
    )

    # --------------------------------------------------------
    # INFERENCE ONLY
    #
    # No optimizer.
    # No backward().
    # No training.
    # --------------------------------------------------------

    with torch.no_grad():

        prediction_scaled = model(
            sequence_tensor,
            future_tensor,
        )

    prediction_scaled = (
        prediction_scaled
        .cpu()
        .numpy()
    )

    prediction = (
        target_scaler
        .inverse_transform(
            prediction_scaled
        )
        .reshape(-1)[0]
    )

    return float(
        prediction
    )


# ============================================================
# MAIN
# ============================================================

def main(args):

    # --------------------------------------------------------
    # Friendly target names
    # --------------------------------------------------------

    if args.target == "traffic":

        target = (
            "Total_Traffic"
        )

    elif args.target == "cash":

        target = (
            "Total_Cash"
        )

    else:

        raise ValueError(
            "target must be traffic or cash"
        )

    key = (
        target,
        args.horizon,
    )

    if key not in FINAL_CONFIGS:

        raise ValueError(
            "Only horizon 1 and 7 are supported."
        )

    config = (
        FINAL_CONFIGS[key]
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PADMA BRIDGE SINGLE FORECAST"
    )

    print(
        "=" * 72
    )

    print(
        f"\nDevice       : {device}"
    )

    print(
        f"Target       : {target}"
    )

    print(
        f"Model        : {config['model']}"
    )

    print(
        f"Window       : {config['window']} days"
    )

    print(
        f"Horizon      : {args.horizon} day(s)"
    )

    print(
        f"Hidden size  : {config['hidden_size']}"
    )

    print(
        f"Dropout      : {config['dropout']}"
    )

    print(
        f"Forecast date: {args.date}"
    )

    # --------------------------------------------------------
    # Build complete feature dataframe
    # --------------------------------------------------------

    df = (
        prepare_timeseries_dataframe()
    )

    (
        sequence,
        future_features,
        actual_value,
        start_date,
        origin_date,
        target_date,
        feature_columns,

    ) = prepare_single_input(

        df=df,

        target_date=
            args.date,

        target=
            target,

        horizon=
            args.horizon,

        window=
            config[
                "window"
            ],
    )

    print(
        "\nINPUT SEQUENCE"
    )

    print(
        "-" * 72
    )

    print(
        f"Starts       : {start_date.date()}"
    )

    print(
        f"Ends         : {origin_date.date()}"
    )

    print(
        f"Ordered days : {sequence.shape[1]}"
    )

    print(
        f"Features/day : {sequence.shape[2]}"
    )

    print(
        f"Tensor shape : {sequence.shape}"
    )

    print(
        "\nImportant:"
    )

    print(
        f"No observations after "
        f"{origin_date.date()} "
        f"are used as historical input."
    )

    # --------------------------------------------------------
    # Three saved models
    # --------------------------------------------------------

    predictions = []

    print(
        "\nSEED PREDICTIONS"
    )

    print(
        "-" * 72
    )

    for seed in SEEDS:

        prediction = (
            predict_one_seed(

                sequence=
                    sequence,

                future_features=
                    future_features,

                feature_columns=
                    feature_columns,

                config=
                    config,

                target=
                    target,

                horizon=
                    args.horizon,

                seed=
                    seed,

                device=
                    device,
            )
        )

        predictions.append(
            prediction
        )

        print(
            f"Seed {seed:>2}: "
            f"{prediction:,.2f}"
        )

    # --------------------------------------------------------
    # Ensemble prediction
    # --------------------------------------------------------

    ensemble_prediction = float(
        np.mean(
            predictions
        )
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FINAL 3-SEED ENSEMBLE FORECAST"
    )

    print(
        "=" * 72
    )

    if target == "Total_Traffic":

        print(
            f"\nPredicted traffic : "
            f"{ensemble_prediction:,.0f} vehicles"
        )

    else:

        print(
            f"\nPredicted toll    : "
            f"{ensemble_prediction:,.2f} BDT"
        )

    # --------------------------------------------------------
    # If actual value exists, show it AFTER prediction.
    # It was never supplied to the model.
    # --------------------------------------------------------

    if actual_value is not None:

        absolute_error = abs(
            actual_value
            -
            ensemble_prediction
        )

        percentage_error = (
            absolute_error
            /
            abs(actual_value)
            *
            100
        )

        print(
            "\nACTUAL VALUE "
            "(for evaluation only)"
        )

        print(
            "-" * 72
        )

        if target == "Total_Traffic":

            print(
                f"Actual traffic    : "
                f"{actual_value:,.0f} vehicles"
            )

            print(
                f"Absolute error    : "
                f"{absolute_error:,.0f} vehicles"
            )

        else:

            print(
                f"Actual toll       : "
                f"{actual_value:,.2f} BDT"
            )

            print(
                f"Absolute error    : "
                f"{absolute_error:,.2f} BDT"
            )

        print(
            f"Percentage error  : "
            f"{percentage_error:.2f}%"
        )

    # --------------------------------------------------------
    # Identify whether this date belongs to test range
    # --------------------------------------------------------

    print(
        "\nDATA SPLIT"
    )

    print(
        "-" * 72
    )

    if (
        target_date
        <= pd.Timestamp(
            "2025-04-22"
        )
    ):

        split_name = "TRAIN"

    elif (
        target_date
        <= pd.Timestamp(
            "2025-12-22"
        )
    ):

        split_name = "VALIDATION"

    else:

        split_name = "TEST"

    print(
        f"This target date belongs to: "
        f"{split_name}"
    )

    # --------------------------------------------------------
    # Save demonstration result
    # --------------------------------------------------------

    output_dir = (
        ROOT
        / "outputs"
        / "live_predictions"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        /
        (
            f"{args.target}"
            f"_h{args.horizon}"
            f"_{target_date.date()}"
            f".csv"
        )
    )

    row = {
        "target": target,
        "model": config["model"],
        "window": config["window"],
        "horizon": args.horizon,

        "input_start_date":
            start_date.date(),

        "input_end_date":
            origin_date.date(),

        "forecast_date":
            target_date.date(),

        "seed_1_prediction":
            predictions[0],

        "seed_7_prediction":
            predictions[1],

        "seed_42_prediction":
            predictions[2],

        "ensemble_prediction":
            ensemble_prediction,

        "actual":
            actual_value,

        "split":
            split_name,
    }

    pd.DataFrame(
        [row]
    ).to_csv(
        output_file,
        index=False
    )

    print(
        "\nSaved demonstration output:"
    )

    print(
        output_file
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        required=True,
        choices=[
            "traffic",
            "cash",
        ],
    )

    parser.add_argument(
        "--horizon",
        required=True,
        type=int,
        choices=[
            1,
            7,
        ],
    )

    parser.add_argument(
        "--date",
        required=True,
        type=str,
        help="Forecast target date in YYYY-MM-DD format.",
    )

    args = parser.parse_args()

    main(
        args
    )