import numpy as np
import pandas as pd


def extract_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Convert a sensor's time-series measurements
    into statistical features for each cycle.
    """

    features = pd.DataFrame(index=df.index)

    features[f"{prefix}_mean"] = df.mean(axis=1)
    features[f"{prefix}_std"] = df.std(axis=1)
    features[f"{prefix}_min"] = df.min(axis=1)
    features[f"{prefix}_max"] = df.max(axis=1)
    features[f"{prefix}_median"] = df.median(axis=1)

    features[f"{prefix}_rms"] = np.sqrt(
        (df ** 2).mean(axis=1)
    )

    return features