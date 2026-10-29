from pathlib import Path

import pandas as pd

from ml.src.features.feature_engineering import extract_features

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


SENSORS = [
    "PS1",
    "PS2",
    "PS3",
    "PS4",
    "PS5",
    "PS6",
    "EPS1",
    "FS1",
    "FS2",
    "TS1",
    "TS2",
    "TS3",
    "TS4",
    "VS1",
    "CE",
    "CP",
    "SE",
]


def load_sensor(sensor: str) -> pd.DataFrame:

    path = RAW_DIR / f"{sensor}.txt"

    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )


def main():

    all_features = []

    for sensor in SENSORS:

        print(f"Processing {sensor}...")

        df = load_sensor(sensor)

        features = extract_features(
            df,
            sensor
        )

        all_features.append(features)

    X = pd.concat(
        all_features,
        axis=1
    )

    profile = pd.read_csv(
        RAW_DIR / "profile.txt",
        sep=r"\s+",
        header=None
    )

    profile.columns = [
        "cooler_condition",
        "valve_condition",
        "pump_leakage",
        "accumulator_pressure",
        "stable_flag",
    ]

    dataset = pd.concat(
        [X, profile],
        axis=1
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = PROCESSED_DIR / "hydraulic_features.csv"

    dataset.to_csv(
        output_path,
        index=False
    )

    print("\nDataset created!")
    print("Shape:", dataset.shape)
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()