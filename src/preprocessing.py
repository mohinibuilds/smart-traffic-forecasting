"""
Data Preprocessing & Feature Engineering
------------------------------------------
- Loads raw traffic CSV
- Handles missing values via temporal interpolation
- Encodes cyclical time features using sine/cosine transformations
- Computes lag features and rolling window statistics
- Scales features for ML / LSTM consumption
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


# ────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ────────────────────────────────────────────────

def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load raw CSV and perform basic cleaning."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df.sort_values(["segment_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Linear interpolation per segment (works without DatetimeIndex)
    numeric_cols = ["vehicle_volume", "avg_speed_kmh", "temperature_c", "rainfall_mm", "visibility_km"]
    for col in numeric_cols:
        df[col] = (
            df.groupby("segment_id")[col]
            .transform(lambda x: x.interpolate(method="linear", limit_direction="both"))
        )
    print(f"[OK] Loaded {len(df):,} rows from '{filepath}'")
    return df


# ────────────────────────────────────────────────
# 2. CYCLICAL ENCODING
# ────────────────────────────────────────────────

def encode_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode hour-of-day and day-of-week as sine/cosine pairs so that
    the model understands the circular (periodic) nature of time.
    """
    df = df.copy()
    # Hour: period = 24
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    # Day of week: period = 7
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    # Month: period = 12
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


# ────────────────────────────────────────────────
# 3. LAG FEATURES & ROLLING STATISTICS
# ────────────────────────────────────────────────

def add_lag_and_rolling_features(df: pd.DataFrame, target_col: str = "vehicle_volume") -> pd.DataFrame:
    """
    For each segment, add:
      - Lag features : t-1, t-2, t-3, t-24 (same hour yesterday)
      - Rolling window stats : 3-h, 6-h, 24-h moving average and std
    """
    df = df.copy()
    lags = [1, 2, 3, 24]
    windows = [3, 6, 24]

    df = df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)

    for lag in lags:
        df[f"{target_col}_lag{lag}"] = (
            df.groupby("segment_id")[target_col].shift(lag)
        )

    for w in windows:
        rolled = df.groupby("segment_id")[target_col].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
        df[f"{target_col}_rollmean{w}"] = rolled

        rolled_std = df.groupby("segment_id")[target_col].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).std().fillna(0)
        )
        df[f"{target_col}_rollstd{w}"] = rolled_std

    # Drop rows with NaN lags (first 24 h per segment)
    df.dropna(subset=[f"{target_col}_lag24"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[OK] Lag & rolling features added. Shape: {df.shape}")
    return df


# ────────────────────────────────────────────────
# 4. FEATURE SELECTION & SCALING
# ────────────────────────────────────────────────

FEATURE_COLS = [
    "vehicle_volume_lag1", "vehicle_volume_lag2", "vehicle_volume_lag3", "vehicle_volume_lag24",
    "vehicle_volume_rollmean3", "vehicle_volume_rollmean6", "vehicle_volume_rollmean24",
    "vehicle_volume_rollstd3", "vehicle_volume_rollstd6", "vehicle_volume_rollstd24",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "temperature_c", "rainfall_mm", "visibility_km",
]
TARGET_COL = "vehicle_volume"


def scale_features(
    df: pd.DataFrame,
    scaler_path: str = "models/saved/feature_scaler.pkl",
    fit: bool = True,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Scale feature matrix using MinMaxScaler.

    Parameters
    ----------
    df : DataFrame with FEATURE_COLS and TARGET_COL present.
    scaler_path : path to persist / load the fitted scaler.
    fit : if True, fit a new scaler; if False, load from disk.

    Returns
    -------
    X_scaled, y, scaler
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    if fit:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)
        print(f"[OK] Scaler fitted & saved → '{scaler_path}'")
    else:
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
        print(f"[OK] Scaler loaded from '{scaler_path}'")

    return X_scaled, y, scaler


# ────────────────────────────────────────────────
# 5. FULL PIPELINE
# ────────────────────────────────────────────────

def build_feature_matrix(
    filepath: str,
    scaler_path: str = "models/saved/feature_scaler.pkl",
    fit_scaler: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, MinMaxScaler]:
    """End-to-end preprocessing pipeline."""
    df = load_and_clean(filepath)
    df = encode_cyclical(df)
    df = add_lag_and_rolling_features(df)
    X_scaled, y, scaler = scale_features(df, scaler_path=scaler_path, fit=fit_scaler)
    return X_scaled, y, df, scaler


if __name__ == "__main__":
    X, y, df_proc, scaler = build_feature_matrix("data/raw/traffic_data.csv")
    print(f"Feature matrix : {X.shape}")
    print(f"Target vector  : {y.shape}")
