from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


# ============================================================
# FINAL RAW TIME-SERIES CONFIGURATIONS
# ============================================================

CONFIGS = [

    # --------------------------------------------------------
    # 1-DAY TRAFFIC
    # --------------------------------------------------------
    {
        "name": "Traffic H1",
        "model": "LSTM",
        "target": "Total_Traffic",
        "window": 30,
        "horizon": 1,
        "hidden_size": 64,
        "dropout": 0.20,
    },

    # --------------------------------------------------------
    # 1-DAY TOLL
    # --------------------------------------------------------
    {
        "name": "Cash H1",
        "model": "LSTM",
        "target": "Total_Cash",
        "window": 14,
        "horizon": 1,
        "hidden_size": 96,
        "dropout": 0.10,
    },

    # --------------------------------------------------------
    # 7-DAY TRAFFIC
    # --------------------------------------------------------
    {
        "name": "Traffic H7",
        "model": "GRU",
        "target": "Total_Traffic",
        "window": 90,
        "horizon": 7,
        "hidden_size": 64,
        "dropout": 0.10,
    },

    # --------------------------------------------------------
    # 7-DAY TOLL
    # --------------------------------------------------------
    {
        "name": "Cash H7",
        "model": "GRU",
        "target": "Total_Cash",
        "window": 60,
        "horizon": 7,
        "hidden_size": 64,
        "dropout": 0.20,
    },
]


SEEDS = [
    1,
    7,
    42,
]


# ============================================================
# TRAIN EVERYTHING
# ============================================================

total_runs = (
    len(CONFIGS)
    * len(SEEDS)
)

run_number = 0


for config in CONFIGS:

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"STARTING: {config['name']}"
    )

    print(
        "=" * 80
    )

    for seed in SEEDS:

        run_number += 1

        print(
            f"\nRUN {run_number}/{total_runs}"
        )

        print(
            f"{config['name']} | Seed {seed}"
        )

        command = [

            sys.executable,

            str(
                ROOT
                / "train_model.py"
            ),

            "--model",
            config["model"],

            "--input_type",
            "raw",

            "--target",
            config["target"],

            "--window",
            str(
                config["window"]
            ),

            "--horizon",
            str(
                config["horizon"]
            ),

            "--hidden_size",
            str(
                config["hidden_size"]
            ),

            "--num_layers",
            "1",

            "--dropout",
            str(
                config["dropout"]
            ),

            "--learning_rate",
            "0.001",

            "--weight_decay",
            "0.0001",

            "--seed",
            str(seed),
        ]

        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
        )


print(
    "\n"
    + "=" * 80
)

print(
    "ALL RAW FINAL MODELS FINISHED"
)

print(
    "=" * 80
)

print(
    f"\nCompleted {total_runs} training runs."
)

print(
    "\nOutputs are in:"
)

print(
    ROOT / "outputs"
)