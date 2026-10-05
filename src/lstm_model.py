"""
LSTM Model for Sequential Traffic Forecasting
-----------------------------------------------
Builds and trains a Long Short-Term Memory (LSTM) neural network
to capture long-term temporal dependencies in traffic time-series data.

Sequence input shape: (batch_size, LOOKBACK, n_features)
"""

import numpy as np
import pandas as pd
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info/warning logs

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ────────────────────────────────────────────────
# SEQUENCE BUILDER
# ────────────────────────────────────────────────

LOOKBACK = 24  # Use last 24 hours as input sequence


def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int = LOOKBACK,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshape flat feature matrix into 3-D sequences for LSTM input.

    Parameters
    ----------
    X : (n_samples, n_features) scaled feature array
    y : (n_samples,) target array
    lookback : number of time steps per sequence

    Returns
    -------
    X_seq : (n_sequences, lookback, n_features)
    y_seq : (n_sequences,)
    """
    n_samples, n_features = X.shape
    X_seq, y_seq = [], []
    for i in range(lookback, n_samples):
        X_seq.append(X[i - lookback: i, :])
        y_seq.append(y[i])
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)
    print(f"[✓] Sequences built: X={X_seq.shape}, y={y_seq.shape}")
    return X_seq, y_seq


# ────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ────────────────────────────────────────────────

def build_lstm_model(
    lookback: int,
    n_features: int,
    units_1: int = 128,
    units_2: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Stacked LSTM with Batch Normalisation and Dropout regularisation.

    Architecture:
      LSTM(128) → BN → Dropout
      LSTM(64)  → BN → Dropout
      Dense(32) → Dense(1)
    """
    model = Sequential(
        [
            LSTM(units_1, return_sequences=True, input_shape=(lookback, n_features)),
            BatchNormalization(),
            Dropout(dropout_rate),
            LSTM(units_2, return_sequences=False),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(32, activation="relu"),
            Dense(1),
        ],
        name="TrafficLSTM",
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()
    return model


# ────────────────────────────────────────────────
# TRAINING
# ────────────────────────────────────────────────

def train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int = LOOKBACK,
    epochs: int = 50,
    batch_size: int = 64,
    validation_split: float = 0.1,
    save_dir: str = "models/saved",
) -> tuple[tf.keras.Model, dict]:
    """
    Build sequences, split data, and train the LSTM model.

    Returns
    -------
    model : trained Keras model
    history_dict : training history metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "lstm_best.keras")

    X_seq, y_seq = build_sequences(X, y, lookback)

    # Time-aware split (no shuffle) to prevent leakage
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    print(f"[*] Train sequences: {len(X_train):,} | Test sequences: {len(X_test):,}")

    model = build_lstm_model(lookback=lookback, n_features=X.shape[1])

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluation on held-out test set ──
    y_pred = model.predict(X_test, verbose=0).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(
        f"\n{'─'*40}\n"
        f"  LSTM Test Results\n"
        f"{'─'*40}\n"
        f"  MAE  : {mae:.2f} vehicles\n"
        f"  RMSE : {rmse:.2f} vehicles\n"
        f"  R²   : {r2:.4f}\n"
    )

    # Save final model
    final_path = os.path.join(save_dir, "lstm_final.keras")
    model.save(final_path)
    print(f"[✓] LSTM model saved → '{final_path}'")

    history_dict = {
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "train_mae": history.history["mae"],
        "val_mae": history.history["val_mae"],
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
        "y_test": y_test,
        "y_pred": y_pred,
    }
    return model, history_dict


# ────────────────────────────────────────────────
# INFERENCE HELPER
# ────────────────────────────────────────────────

def predict_lstm(
    model_path: str,
    X_sequence: np.ndarray,
) -> np.ndarray:
    """
    Load a saved LSTM model and run inference on a pre-built sequence array.

    Parameters
    ----------
    model_path : path to saved .keras model file
    X_sequence : (n, lookback, n_features) float32 array

    Returns
    -------
    predictions : (n,) float32 array of predicted vehicle volumes
    """
    model = load_model(model_path)
    preds = model.predict(X_sequence, verbose=0).flatten()
    return preds


if __name__ == "__main__":
    from preprocessing import build_feature_matrix

    X, y, df, scaler = build_feature_matrix("data/raw/traffic_data.csv")
    model, history = train_lstm(X, y, epochs=10)  # quick test run
