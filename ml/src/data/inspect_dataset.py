from pathlib import Path
import pandas as pd


RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def inspect_file(filename: str):
    path = RAW_DATA_DIR / filename

    print(f"\n===== {filename} =====")
    print(f"Path: {path}")
    print(f"Exists: {path.exists()}")

    if not path.exists():
        return

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )

    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    inspect_file("profile.txt")
    inspect_file("PS1.txt")
    inspect_file("TS1.txt")