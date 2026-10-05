"""
Evaluation & Visualization Utilities
--------------------------------------
Provides functions to:
  - Plot actual vs. predicted traffic volumes
  - Plot LSTM training/validation loss curves
  - Generate confusion-style congestion severity classification report
  - Plot feature importances
  - Generate Folium heatmap for road segment congestion
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# Optional folium for map rendering
try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
PALETTE = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}


# ────────────────────────────────────────────────
# 1. ACTUAL vs PREDICTED
# ────────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    n_points: int = 500,
    save_path: str = None,
) -> plt.Figure:
    """Plot actual vs. predicted traffic volume over time (first n_points)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # ── Time series comparison ──
    ax = axes[0]
    x = np.arange(min(n_points, len(y_true)))
    ax.plot(x, y_true[:n_points], label="Actual", color="#2c3e50", linewidth=1.2, alpha=0.85)
    ax.plot(x, y_pred[:n_points], label="Predicted", color="#3b82d4", linewidth=1.2, linestyle="--")
    ax.set_title(f"{model_name} — Actual vs. Predicted")
    ax.set_xlabel("Time Step (hours)")
    ax.set_ylabel("Vehicle Volume")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── Scatter plot ──
    ax2 = axes[1]
    ax2.scatter(y_true[:n_points], y_pred[:n_points], alpha=0.3, s=15, color="#3b82d4")
    lim = [min(y_true.min(), y_pred.min()) * 0.95, max(y_true.max(), y_pred.max()) * 1.05]
    ax2.plot(lim, lim, "r--", linewidth=1.5, label="Perfect fit")
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_xlabel("Actual Volume")
    ax2.set_ylabel("Predicted Volume")
    ax2.set_title(f"{model_name} — Prediction Scatter")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[✓] Plot saved → '{save_path}'")
    return fig


# ────────────────────────────────────────────────
# 2. LSTM TRAINING CURVES
# ────────────────────────────────────────────────

def plot_training_curves(
    history_dict: dict,
    save_path: str = None,
) -> plt.Figure:
    """Plot LSTM training & validation loss + MAE curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric, label in zip(
        axes,
        [("train_loss", "val_loss"), ("train_mae", "val_mae")],
        ["Loss (MSE)", "MAE"],
    ):
        ax.plot(history_dict[metric[0]], label="Train", color="#3b82d4")
        ax.plot(history_dict[metric[1]], label="Validation", color="#e74c3c", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"LSTM Training — {label}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[✓] Training curves saved → '{save_path}'")
    return fig


# ────────────────────────────────────────────────
# 3. FEATURE IMPORTANCE BAR CHART
# ────────────────────────────────────────────────

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    save_path: str = None,
) -> plt.Figure:
    """Horizontal bar chart for top-N feature importances."""
    top = importance_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#3b82d4" if i < 3 else "#7c5cd8" if i < 7 else "#57606a" for i in range(len(top))]
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1], edgecolor="none")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[✓] Feature importance chart saved → '{save_path}'")
    return fig


# ────────────────────────────────────────────────
# 4. CONGESTION SEVERITY DISTRIBUTION
# ────────────────────────────────────────────────

def plot_severity_distribution(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Pie + bar chart of congestion severity class distribution."""
    counts = df["congestion_severity"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Pie
    axes[0].pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=[PALETTE.get(k, "#aaa") for k in counts.index],
        startangle=90,
    )
    axes[0].set_title("Congestion Severity Distribution")

    # Bar
    axes[1].bar(
        counts.index,
        counts.values,
        color=[PALETTE.get(k, "#aaa") for k in counts.index],
        edgecolor="none",
    )
    axes[1].set_ylabel("Count")
    axes[1].set_title("Severity Counts per Class")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[✓] Severity distribution saved → '{save_path}'")
    return fig


# ────────────────────────────────────────────────
# 5. FOLIUM TRAFFIC HEATMAP
# ────────────────────────────────────────────────

# Approximate city-centre coordinates for simulated segments
SEGMENT_COORDS = {
    1:  (40.7128, -74.0060),   # New York
    2:  (34.0522, -118.2437),  # Los Angeles
    3:  (41.8781, -87.6298),   # Chicago
    4:  (29.7604, -95.3698),   # Houston
    5:  (33.4484, -112.0740),  # Phoenix
    6:  (39.9526, -75.1652),   # Philadelphia
    7:  (29.4241, -98.4936),   # San Antonio
    8:  (32.7767, -96.7970),   # Dallas
    9:  (30.3322, -81.6557),   # Jacksonville
    10: (30.2672, -97.7431),   # Austin
}


def generate_folium_heatmap(
    df_snapshot: pd.DataFrame,
    output_path: str = "assets/traffic_heatmap.html",
) -> str:
    """
    Generate a Folium-based interactive heatmap for a snapshot DataFrame.

    Parameters
    ----------
    df_snapshot : DataFrame with columns ['segment_id', 'vehicle_volume']
                  (typically a single timestamp slice)
    output_path : path to save the HTML map

    Returns
    -------
    output_path : path of the saved HTML file
    """
    if not FOLIUM_AVAILABLE:
        raise ImportError("folium is not installed. Run: pip install folium")

    m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)

    heat_data = []
    for _, row in df_snapshot.iterrows():
        seg = int(row["segment_id"])
        vol = float(row["vehicle_volume"])
        if seg in SEGMENT_COORDS:
            lat, lon = SEGMENT_COORDS[seg]
            heat_data.append([lat, lon, vol])

    HeatMap(heat_data, radius=30, blur=20, max_zoom=10).add_to(m)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    m.save(output_path)
    try:
        print(f"[✓] Folium heatmap saved → '{output_path}'")
    except (ValueError, OSError):
        pass
    return output_path


# ────────────────────────────────────────────────
# 6. METRICS COMPARISON TABLE
# ────────────────────────────────────────────────

def print_metrics_table(results: dict) -> pd.DataFrame:
    """Print a formatted comparison table of model evaluation metrics."""
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": r.get("model", name),
            "MAE": round(r["MAE"], 2),
            "RMSE": round(r["RMSE"], 2),
            "R²": round(r["R2"], 4),
        })
    df = pd.DataFrame(rows)
    print("\n" + "=" * 45)
    print("  Model Comparison Summary")
    print("=" * 45)
    print(df.to_string(index=False))
    print("=" * 45 + "\n")
    return df
