"""
Baseline ML Models — XGBoost & Random Forest
----------------------------------------------
Trains, evaluates, and persists tabular ML models for short-term
traffic volume forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[!] XGBoost not installed. Only RandomForest will be available.")


# ────────────────────────────────────────────────
# EVALUATION HELPER
# ────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    """Compute and display MAE, RMSE, R² metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    metrics = {"model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}
    print(
        f"\n{'─'*40}\n"
        f"  {model_name} Results\n"
        f"{'─'*40}\n"
        f"  MAE  : {mae:.2f} vehicles\n"
        f"  RMSE : {rmse:.2f} vehicles\n"
        f"  R²   : {r2:.4f}\n"
    )
    return metrics


# ────────────────────────────────────────────────
# RANDOM FOREST
# ────────────────────────────────────────────────

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 15,
    n_jobs: int = -1,
    save_path: str = "models/saved/random_forest.pkl",
) -> RandomForestRegressor:
    """Train a Random Forest Regressor and save it to disk."""
    print("[*] Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"[OK] Random Forest saved → '{save_path}'")
    return model


# ────────────────────────────────────────────────
# XGBOOST
# ────────────────────────────────────────────────

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 7,
    save_path: str = "models/saved/xgboost.pkl",
) -> "XGBRegressor":
    """Train an XGBoost Regressor and save it to disk."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")
    print("[*] Training XGBoost...")
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"[OK] XGBoost saved → '{save_path}'")
    return model


# ────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ────────────────────────────────────────────────

def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Return a sorted DataFrame of feature importances."""
    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df.sort_values("importance", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ────────────────────────────────────────────────
# TRAIN & EVALUATE PIPELINE
# ────────────────────────────────────────────────

def run_baseline_training(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    test_size: float = 0.2,
) -> dict:
    """
    Split data, train both models, evaluate, and return results summary.
    Uses a time-aware split (no shuffling) to avoid data leakage.
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n[*] Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")

    results = {}

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    results["RandomForest"] = evaluate(y_test, rf_preds, "Random Forest")
    results["RandomForest"]["model_obj"] = rf_model
    results["RandomForest"]["predictions"] = rf_preds
    results["RandomForest"]["y_test"] = y_test

    # XGBoost
    if XGBOOST_AVAILABLE:
        xgb_model = train_xgboost(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        results["XGBoost"] = evaluate(y_test, xgb_preds, "XGBoost")
        results["XGBoost"]["model_obj"] = xgb_model
        results["XGBoost"]["predictions"] = xgb_preds
        results["XGBoost"]["y_test"] = y_test

        # Feature importance from XGBoost
        fi_df = get_feature_importance(xgb_model, feature_names)
        print("\nTop 10 Features (XGBoost):")
        print(fi_df.head(10).to_string(index=False))

    return results


if __name__ == "__main__":
    from preprocessing import build_feature_matrix, FEATURE_COLS

    X, y, df, scaler = build_feature_matrix("data/raw/traffic_data.csv")
    run_baseline_training(X, y, feature_names=FEATURE_COLS)
