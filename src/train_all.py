"""
Training Orchestrator
----------------------
End-to-end script that:
  1. Generates synthetic data (if no CSV found)
  2. Runs the full preprocessing pipeline
  3. Trains the Random Forest and XGBoost baseline models
  4. Trains the LSTM sequential model
  5. Prints a final comparison table

Run:
  python src/train_all.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Ensure src directory is on the path ──────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "traffic_data.csv")
SCALER_PATH = os.path.join(ROOT_DIR, "models", "saved", "feature_scaler.pkl")
MODELS_DIR = os.path.join(ROOT_DIR, "models", "saved")


def main():
    print("\n" + "=" * 55)
    print("  Smart Traffic Forecasting — Training Pipeline")
    print("=" * 55)

    # ── Step 1: Data generation ──────────────────
    if not os.path.exists(DATA_PATH):
        print("\n[1/4] Generating synthetic traffic dataset...")
        from data_generator import generate_traffic_dataset
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        generate_traffic_dataset(output_path=DATA_PATH)
    else:
        print(f"\n[1/4] Dataset already exists at '{DATA_PATH}' — skipping generation.")

    # ── Step 2: Preprocessing ────────────────────
    print("\n[2/4] Running preprocessing pipeline...")
    from preprocessing import build_feature_matrix, FEATURE_COLS
    X, y, df, scaler = build_feature_matrix(
        filepath=DATA_PATH,
        scaler_path=SCALER_PATH,
        fit_scaler=True,
    )
    print(f"  Feature matrix : {X.shape}")
    print(f"  Target vector  : {y.shape}")

    # ── Step 3: Baseline models ──────────────────
    print("\n[3/4] Training baseline models (RF + XGBoost)...")
    from baseline_models import run_baseline_training
    results = run_baseline_training(X, y, feature_names=FEATURE_COLS)

    # ── Step 4: LSTM (requires Python ≤ 3.11) ────
    print("\n[4/4] Training LSTM model...")
    try:
        from lstm_model import train_lstm
        lstm_model, lstm_history = train_lstm(
            X, y,
            epochs=30,
            batch_size=64,
            save_dir=MODELS_DIR,
        )
        results["LSTM"] = {
            "model": "LSTM",
            "MAE": lstm_history["test_mae"],
            "RMSE": lstm_history["test_rmse"],
            "R2": lstm_history["test_r2"],
        }
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[!] LSTM skipped — TensorFlow not available on this Python version: {e}")
        print("    TensorFlow requires Python 3.11 or lower.")
        print("    RF and XGBoost models are fully trained and ready.")

    # ── Final comparison table ───────────────────
    from evaluation import print_metrics_table
    print_metrics_table(results)

    print("\n[OK] Training complete. Launch the dashboard with:")
    print("      python -m streamlit run app.py\n")


if __name__ == "__main__":
    main()
