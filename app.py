"""
Streamlit Dashboard — Smart Urban Traffic Forecasting
=======================================================
Run with:  streamlit run app.py

Features:
  - Sidebar controls for segment selection, forecast horizon, and model choice
  - Live congestion severity alerts (Low / Medium / High)
  - 2–4 hour ahead forecast with confidence bands
  - Historical volume trend chart
  - Feature importance visualization
  - Interactive Folium traffic heatmap embed
  - Model metrics comparison table
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ── Path setup: make both the project root and src/ importable ──
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_ROOT, "src")
for _p in [_ROOT, _SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Change working directory to project root so relative file paths work ──
os.chdir(_ROOT)

# ── Page config ──────────────────────────────────
st.set_page_config(
    page_title="Smart Traffic Forecasting",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .severity-low    { color: #2ecc71; font-weight: 700; font-size: 1.4rem; }
    .severity-medium { color: #f39c12; font-weight: 700; font-size: 1.4rem; }
    .severity-high   { color: #e74c3c; font-weight: 700; font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────
# DATA & MODEL CACHING
# ────────────────────────────────────────────────

DATA_PATH   = os.path.join(_ROOT, "data", "raw", "traffic_data.csv")
SCALER_PATH = os.path.join(_ROOT, "models", "saved", "feature_scaler.pkl")
RF_PATH     = os.path.join(_ROOT, "models", "saved", "random_forest.pkl")
XGB_PATH    = os.path.join(_ROOT, "models", "saved", "xgboost.pkl")


def _ensure_models_exist():
    """Auto-train all models if they are not present (first cloud run)."""
    if not os.path.exists(RF_PATH) or not os.path.exists(SCALER_PATH):
        import sys
        sys.path.insert(0, _SRC)
        with st.spinner("⚙️ First run — generating data & training models (2–3 min)..."):
            # Generate data if needed
            if not os.path.exists(DATA_PATH):
                from data_generator import generate_traffic_dataset
                os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
                generate_traffic_dataset(output_path=DATA_PATH)

            # Preprocess
            from preprocessing import build_feature_matrix, FEATURE_COLS
            X, y, _, _ = build_feature_matrix(
                filepath=DATA_PATH,
                scaler_path=SCALER_PATH,
                fit_scaler=True,
            )

            # Train RF + XGBoost
            from baseline_models import run_baseline_training
            run_baseline_training(X, y, feature_names=FEATURE_COLS)

        st.success("✅ Models trained! Loading dashboard...")
        st.rerun()


@st.cache_data(show_spinner="Loading traffic dataset...")
def load_data():
    from preprocessing import load_and_clean, encode_cyclical, add_lag_and_rolling_features

    if not os.path.exists(DATA_PATH):
        from data_generator import generate_traffic_dataset
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        generate_traffic_dataset(output_path=DATA_PATH)

    df = load_and_clean(DATA_PATH)
    df = encode_cyclical(df)
    df = add_lag_and_rolling_features(df)
    return df


@st.cache_resource(show_spinner="Loading ML models...")
def load_models():
    import joblib
    models = {}
    if os.path.exists(RF_PATH):
        models["Random Forest"] = joblib.load(RF_PATH)
    if os.path.exists(XGB_PATH):
        models["XGBoost"] = joblib.load(XGB_PATH)
    return models


@st.cache_resource(show_spinner="Loading scaler...")
def load_scaler():
    import joblib
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None


# ────────────────────────────────────────────────
# HELPER FUNCTIONS
# ────────────────────────────────────────────────

FEATURE_COLS = [
    "vehicle_volume_lag1", "vehicle_volume_lag2", "vehicle_volume_lag3", "vehicle_volume_lag24",
    "vehicle_volume_rollmean3", "vehicle_volume_rollmean6", "vehicle_volume_rollmean24",
    "vehicle_volume_rollstd3", "vehicle_volume_rollstd6", "vehicle_volume_rollstd24",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "temperature_c", "rainfall_mm", "visibility_km",
]


def classify_severity(volume: float, capacity: float = 1200) -> str:
    ratio = volume / capacity
    if ratio < 0.6:
        return "Low"
    elif ratio < 1.0:
        return "Medium"
    return "High"


def severity_badge(severity: str) -> str:
    css_class = f"severity-{severity.lower()}"
    icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(severity, "⚪")
    return f'<span class="{css_class}">{icon} {severity}</span>'


def forecast_ahead(
    df_seg: pd.DataFrame,
    model,
    scaler,
    model_name: str,
    horizon: int = 4,
) -> pd.DataFrame:
    """Generate rolling multi-step ahead forecasts for a single segment."""
    from src.preprocessing import FEATURE_COLS as FC

    # Use the last `horizon` available rows as base
    recent = df_seg.tail(horizon + 24).copy().reset_index(drop=True)
    forecasts = []

    for h in range(horizon):
        row = recent.iloc[-(horizon - h)]
        X_row = row[FC].values.reshape(1, -1)
        X_scaled = scaler.transform(X_row)

        if model_name == "LSTM":
            # For LSTM: build a 24-step sequence ending at the current row
            seq_rows = recent.iloc[max(0, len(recent) - (horizon - h) - 24): len(recent) - (horizon - h)]
            if len(seq_rows) < 24:
                seq_rows = pd.concat([seq_rows.iloc[:1]] * (24 - len(seq_rows)) + [seq_rows])
            X_seq = scaler.transform(seq_rows[FC].values)[-24:]
            pred = model.predict(X_seq[np.newaxis, :, :], verbose=0)[0][0]
        else:
            pred = model.predict(X_scaled)[0]

        ts = recent["timestamp"].iloc[-(horizon - h)] + pd.Timedelta(hours=h + 1)
        forecasts.append({
            "timestamp": ts,
            "predicted_volume": max(0, round(float(pred))),
            "severity": classify_severity(float(pred)),
        })

    return pd.DataFrame(forecasts)


# ────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/SDG_wheel_transparent.png/200px-SDG_wheel_transparent.png",
    width=120,
)
st.sidebar.title("🚦 Traffic Forecasting")
st.sidebar.markdown("**SDG 11** — Sustainable Cities & Communities")
st.sidebar.markdown("---")

segment_id = st.sidebar.selectbox(
    "🛣️ Road Segment",
    options=list(range(1, 11)),
    format_func=lambda x: f"Segment {x}",
)
forecast_horizon = st.sidebar.slider("⏱️ Forecast Horizon (hours)", min_value=1, max_value=4, value=2)
model_choice = st.sidebar.radio("🤖 Prediction Model", ["Random Forest", "XGBoost"])
show_heatmap = st.sidebar.checkbox("🗺️ Show Congestion Heatmap", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "Predictions represent vehicle volume counts for the selected segment "
    "1–4 hours into the future based on recent traffic patterns."
)

# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────

st.title("🏙️ Smart Urban Traffic Congestion & Resource Demand Forecasting")
st.markdown(
    "> **SDG 11.2** — Provide access to safe, affordable, accessible and sustainable transport systems. "
    "Reduce congestion, emissions, and improve urban mobility through data-driven forecasting."
)

# ── Auto-train on first cloud run if models missing ──
_ensure_models_exist()

# ── Load data ──
df = load_data()
models = load_models()
scaler = load_scaler()

if scaler is None:
    st.error("⚠️ Models could not be loaded. Please refresh the page.")
    st.stop()

# ── Filter by segment ──
df_seg = df[df["segment_id"] == segment_id].copy()
df_seg = df_seg.sort_values("timestamp").reset_index(drop=True)

# ── KPI header row ──
latest = df_seg.iloc[-1]
current_vol = int(latest["vehicle_volume"])
current_speed = round(latest["avg_speed_kmh"], 1)
current_severity = classify_severity(current_vol)
current_rain = round(latest["rainfall_mm"], 1)

st.markdown("### 📊 Current Conditions — Segment {seg}".format(seg=segment_id))
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🚗 Vehicle Volume", f"{current_vol:,} veh/hr")
with col2:
    st.metric("⚡ Avg Speed", f"{current_speed} km/h")
with col3:
    st.metric("🌧️ Rainfall", f"{current_rain} mm")
with col4:
    st.markdown(
        f'<div class="metric-card">Congestion<br>{severity_badge(current_severity)}</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── FORECAST SECTION ──
st.markdown(f"### 🔮 {forecast_horizon}-Hour Ahead Forecast")

if model_choice not in models:
    st.warning(
        f"⚠️ **{model_choice}** model not found. "
        "Run `python src/train_all.py` to train and save models."
    )
else:
    model = models[model_choice]
    forecast_df = forecast_ahead(df_seg, model, scaler, model_choice, forecast_horizon)

    # Display forecast cards
    f_cols = st.columns(forecast_horizon)
    for i, row in forecast_df.iterrows():
        with f_cols[i]:
            ts_label = row["timestamp"].strftime("%H:%M")
            st.markdown(
                f"""
                <div class="metric-card">
                  <b>+{i+1}h &nbsp;{ts_label}</b><br>
                  <span style="font-size:1.3rem; font-weight:700;">{row['predicted_volume']:,}</span>
                  <span style="font-size:0.8rem; color:#57606a;"> veh/hr</span><br>
                  {severity_badge(row['severity'])}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")

    # Forecast bar chart
    fig_f, ax_f = plt.subplots(figsize=(8, 3))
    colors_f = [
        "#2ecc71" if s == "Low" else "#f39c12" if s == "Medium" else "#e74c3c"
        for s in forecast_df["severity"]
    ]
    ax_f.bar(
        forecast_df["timestamp"].dt.strftime("%H:%M"),
        forecast_df["predicted_volume"],
        color=colors_f,
        edgecolor="none",
        width=0.5,
    )
    ax_f.axhline(current_vol, color="#3b82d4", linestyle="--", linewidth=1.2, label="Current volume")
    ax_f.set_ylabel("Predicted Vehicle Volume")
    ax_f.set_title(f"Forecast — Segment {segment_id} ({model_choice})")
    ax_f.legend()
    ax_f.grid(axis="y", linestyle="--", alpha=0.4)
    fig_f.tight_layout()
    st.pyplot(fig_f, use_container_width=True)
    plt.close(fig_f)

st.markdown("---")

# ── HISTORICAL TREND ──
st.markdown("### 📈 Historical Traffic Volume (Last 7 Days)")
last_7d = df_seg[df_seg["timestamp"] >= df_seg["timestamp"].max() - pd.Timedelta(days=7)]

fig_h, ax_h = plt.subplots(figsize=(12, 3.5))
ax_h.plot(last_7d["timestamp"], last_7d["vehicle_volume"], color="#2c3e50", linewidth=1.2, alpha=0.85)
ax_h.fill_between(last_7d["timestamp"], last_7d["vehicle_volume"], alpha=0.08, color="#3b82d4")
ax_h.set_ylabel("Vehicle Volume (veh/hr)")
ax_h.set_title(f"Segment {segment_id} — Last 7 Days")
ax_h.grid(True, linestyle="--", alpha=0.4)
fig_h.autofmt_xdate()
fig_h.tight_layout()
st.pyplot(fig_h, use_container_width=True)
plt.close(fig_h)

st.markdown("---")

# ── PEAK HOUR ANALYSIS ──
st.markdown("### 🕐 Average Volume by Hour of Day")
hourly_avg = df_seg.groupby("hour")["vehicle_volume"].mean().reset_index()

fig_p, ax_p = plt.subplots(figsize=(10, 3))
bars = ax_p.bar(hourly_avg["hour"], hourly_avg["vehicle_volume"], edgecolor="none", width=0.7)
# Colour peaks
for bar, h in zip(bars, hourly_avg["hour"]):
    if h in [7, 8, 9, 17, 18, 19]:
        bar.set_color("#e74c3c")
    else:
        bar.set_color("#3b82d4")

ax_p.set_xlabel("Hour of Day")
ax_p.set_ylabel("Avg Vehicle Volume")
ax_p.set_title(f"Segment {segment_id} — Hourly Traffic Profile (🔴 = peak hours)")
ax_p.set_xticks(range(24))
ax_p.grid(axis="y", linestyle="--", alpha=0.4)
fig_p.tight_layout()
st.pyplot(fig_p, use_container_width=True)
plt.close(fig_p)

st.markdown("---")

# ── CONGESTION HEATMAP ──
if show_heatmap:
    st.markdown("### 🗺️ Network-Wide Congestion Heatmap")
    try:
        from src.evaluation import generate_folium_heatmap
        latest_snapshot = (
            df.sort_values("timestamp")
            .groupby("segment_id")
            .last()
            .reset_index()[["segment_id", "vehicle_volume"]]
        )
        map_path = "assets/traffic_heatmap.html"
        os.makedirs("assets", exist_ok=True)
        generate_folium_heatmap(latest_snapshot, output_path=map_path)
        with open(map_path, "r", encoding="utf-8") as f:
            map_html = f.read()
        st.components.v1.html(map_html, height=450)
    except ImportError:
        st.info("Install `folium` to enable the interactive heatmap: `pip install folium`")
    st.markdown("---")

# ── RE-ROUTING RECOMMENDATIONS ──
st.markdown("### 🔀 Re-routing & Resource Allocation Recommendations")

has_high = False
if model_choice in models:
    for _, row in forecast_df.iterrows():
        if row["severity"] == "High":
            has_high = True
            st.error(
                f"🔴 **High Congestion Predicted** at `{row['timestamp'].strftime('%H:%M')}` "
                f"— Forecast volume: **{row['predicted_volume']:,} veh/hr**\n\n"
                "**Recommended actions:**\n"
                "- Activate dynamic signal timing adjustment on adjoining intersections\n"
                "- Divert secondary routes (Ring Road / Bypass) via VMS boards\n"
                "- Pre-position traffic enforcement officers at choke-points\n"
                "- Alert public transit authority to increase bus/metro frequency"
            )
        elif row["severity"] == "Medium":
            st.warning(
                f"🟡 **Moderate Congestion** at `{row['timestamp'].strftime('%H:%M')}` "
                f"— Forecast volume: **{row['predicted_volume']:,} veh/hr**\n\n"
                "**Recommended actions:**\n"
                "- Monitor key junctions for spillback\n"
                "- Communicate advisory travel times on smart boards"
            )

if not has_high and model_choice in models:
    st.success(
        "🟢 **Traffic conditions are expected to remain manageable** over the selected forecast window.\n\n"
        "No emergency re-routing required. Continue standard signal operations."
    )

st.markdown("---")

# ── FOOTER ──
st.markdown(
    """
    <div style="text-align:center; color:#57606a; font-size:0.82rem; margin-top:2rem; 
                border-top:1px solid #e5e7eb; padding-top:1rem;">
      Smart Urban Traffic Forecasting System &nbsp;|&nbsp;
      SDG 11 — Sustainable Cities &amp; Communities &nbsp;|&nbsp;
      Built with Python · scikit-learn · TensorFlow · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
