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

# Fix Windows encoding so ✓ and other unicode chars don't crash the app
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

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

# ── Global CSS (login + dashboard) ───────────────
st.markdown(
    """
    <style>
    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {
        background: #0f1117;
        color: #e6edf3;
        font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    [data-testid="stHeader"] { background: transparent !important; }

    /* ── Login card ── */
    .login-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 80vh;
    }
    .login-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 48px 52px;
        width: 100%;
        max-width: 420px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        text-align: center;
    }
    .login-logo {
        font-size: 3rem;
        margin-bottom: 4px;
    }
    .login-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e6edf3;
        margin: 0 0 4px;
    }
    .login-subtitle {
        font-size: 0.85rem;
        color: #8b949e;
        margin-bottom: 32px;
    }
    .login-error {
        background: rgba(231,76,60,0.15);
        border: 1px solid rgba(231,76,60,0.4);
        border-radius: 8px;
        padding: 10px 14px;
        color: #e74c3c;
        font-size: 0.88rem;
        margin-bottom: 16px;
    }

    /* ── Top header bar ── */
    .top-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 24px;
        margin-bottom: 24px;
    }
    .top-bar-left { display: flex; align-items: center; gap: 14px; }
    .top-bar-logo { font-size: 2rem; }
    .top-bar-title { font-size: 1.25rem; font-weight: 700; color: #e6edf3; margin: 0; }
    .top-bar-sub   { font-size: 0.78rem; color: #8b949e; margin: 0; }
    .top-bar-badge {
        background: #1f6feb;
        color: #fff;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
    }

    /* ── Metric card ── */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
    }
    .severity-low    { color: #2ecc71; font-weight: 700; font-size: 1.4rem; }
    .severity-medium { color: #f39c12; font-weight: 700; font-size: 1.4rem; }
    .severity-high   { color: #e74c3c; font-weight: 700; font-size: 1.4rem; }

    /* ── Streamlit metric overrides ── */
    [data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 18px;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.82rem; }
    [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.4rem; font-weight: 700; }

    /* ── Sidebar inputs ── */
    section[data-testid="stSidebar"] input[type="text"] {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        color: #e6edf3 !important;
        font-size: 0.92rem !important;
    }
    section[data-testid="stSidebar"] input[type="text"]:focus {
        border-color: #1f6feb !important;
        box-shadow: 0 0 0 3px rgba(31,111,235,0.25) !important;
    }

    /* ── Divider ── */
    hr { border-color: #30363d !important; }

    /* ── Matplotlib charts — dark bg ── */
    .stPlotlyChart, .stPyplot { background: transparent !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────
# LOGIN PAGE
# ────────────────────────────────────────────────

# ── User store backed by a JSON file ──
import json

_USERS_FILE = os.path.join(_ROOT, "data", "users.json")
_DEFAULT_USERS = {
    "admin":   "traffic123",
    "analyst": "sdg11@2025",
    "demo":    "demo",
}

def _load_users() -> dict:
    """Load users from file, seeding defaults if file doesn't exist."""
    if os.path.exists(_USERS_FILE):
        try:
            with open(_USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # File missing or corrupt — create it with defaults
    os.makedirs(os.path.dirname(_USERS_FILE), exist_ok=True)
    with open(_USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(_DEFAULT_USERS, f, indent=2)
    return dict(_DEFAULT_USERS)

def _save_users(db: dict):
    os.makedirs(os.path.dirname(_USERS_FILE), exist_ok=True)
    with open(_USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "login_user" not in st.session_state:
    st.session_state.login_user = ""

def _do_login(username: str, password: str):
    uname = username.strip().lower()
    db = _load_users()
    if db.get(uname) == password:
        st.session_state.authenticated = True
        st.session_state.login_user = uname
        st.session_state.pop("login_error", None)
    else:
        st.session_state.login_error = "Invalid username or password."

def _do_register(username: str, password: str, confirm: str):
    uname = username.strip().lower()
    db = _load_users()
    if not uname or not password:
        st.session_state.signup_error = "Username and password cannot be empty."
    elif uname in db:
        st.session_state.signup_error = "Username already exists. Please choose another."
    elif password != confirm:
        st.session_state.signup_error = "Passwords do not match."
    elif len(password) < 6:
        st.session_state.signup_error = "Password must be at least 6 characters."
    else:
        db[uname] = password
        _save_users(db)
        st.session_state.signup_success = True
        st.session_state.pop("signup_error", None)

def _do_logout():
    st.session_state.authenticated = False
    st.session_state.login_user = ""

if not st.session_state.authenticated:
    # Hide sidebar on auth screen
    st.markdown(
        "<style>section[data-testid='stSidebar']{display:none}</style>",
        unsafe_allow_html=True,
    )

    with st.container():
        _, col, _ = st.columns([1, 1.6, 1])
        with col:
            # Logo / brand header
            st.markdown(
                """
                <div style="text-align:center;margin-bottom:8px;">
                  <div style="font-size:3rem;">🚦</div>
                  <p style="font-size:1.45rem;font-weight:700;color:#e6edf3;margin:0;">Smart Traffic Forecasting</p>
                  <p style="font-size:0.82rem;color:#8b949e;margin:0 0 24px;">SDG 11 — Sustainable Cities &amp; Communities</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── After successful registration: show sign-in form pre-filled ──
            if st.session_state.get("signup_success"):
                new_uname = st.session_state.get("new_username", "")
                st.success(f"Account **{new_uname}** created successfully! Sign in below.")

                if st.session_state.get("login_error"):
                    st.error(st.session_state.login_error)
                    st.session_state.pop("login_error", None)

                si_user = st.text_input("Username", value=new_uname, key="si_user_post")
                si_pass = st.text_input("Password", placeholder="Enter your password", type="password", key="si_pass_post")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Sign In", use_container_width=True, type="primary", key="btn_signin_post"):
                    _do_login(si_user, si_pass)
                    if st.session_state.authenticated:
                        st.session_state.signup_success = False
                        st.session_state.pop("new_username", None)
                    st.rerun()

                if st.button("Back to Sign In", use_container_width=False, key="btn_back"):
                    st.session_state.signup_success = False
                    st.session_state.pop("new_username", None)
                    st.rerun()

            else:
                # Normal tab switcher
                tab_signin, tab_signup = st.tabs(["Sign In", "Create Account"])

                # ── Sign In tab ──
                with tab_signin:
                    if st.session_state.get("login_error"):
                        st.error(st.session_state.login_error)
                        st.session_state.pop("login_error", None)

                    si_user = st.text_input("Username", placeholder="Enter your username", key="si_user")
                    si_pass = st.text_input("Password", placeholder="Enter your password", type="password", key="si_pass")
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Sign In", use_container_width=True, type="primary", key="btn_signin"):
                        _do_login(si_user, si_pass)
                        st.rerun()

                    st.markdown(
                        "<p style='color:#8b949e;font-size:0.76rem;margin-top:12px;text-align:center;'>"
                        "Demo: <b>admin</b> / <b>traffic123</b></p>",
                        unsafe_allow_html=True,
                    )

                # ── Create Account tab ──
                with tab_signup:
                    if st.session_state.get("signup_error"):
                        st.error(st.session_state.signup_error)
                        st.session_state.pop("signup_error", None)

                    su_user    = st.text_input("Choose a Username", placeholder="e.g. john_doe", key="su_user")
                    su_pass    = st.text_input("Password", placeholder="Min 6 characters", type="password", key="su_pass")
                    su_confirm = st.text_input("Confirm Password", placeholder="Re-enter password", type="password", key="su_confirm")
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Create Account", use_container_width=True, type="primary", key="btn_signup"):
                        _do_register(su_user, su_pass, su_confirm)
                        if st.session_state.get("signup_success"):
                            st.session_state.new_username = su_user.strip().lower()
                        st.rerun()

                    st.markdown(
                        "<p style='color:#8b949e;font-size:0.76rem;margin-top:12px;text-align:center;'>"
                        "Your account is saved for future logins.</p>",
                        unsafe_allow_html=True,
                    )

    st.stop()

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

        st.success("Models trained! Loading dashboard...")
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
# WORLD LOCATION DATABASE
# Each entry: (Country, State/Province, City, Representative Segment 1-10, Key Road/District)
# ────────────────────────────────────────────────

WORLD_LOCATIONS = [
    # ── INDIA ──
    ("India", "Maharashtra", "Mumbai",      1, "Western Express Highway"),
    ("India", "Maharashtra", "Pune",        2, "FC Road / Hinjewadi"),
    ("India", "Maharashtra", "Nagpur",      3, "Wardha Road"),
    ("India", "Maharashtra", "Nashik",      4, "Mumbai-Agra Highway"),
    ("India", "Delhi",       "New Delhi",   5, "Connaught Place / NH-48"),
    ("India", "Delhi",       "Gurugram",    6, "Golf Course Road"),
    ("India", "Karnataka",   "Bengaluru",   7, "MG Road / ORR"),
    ("India", "Karnataka",   "Mysuru",      8, "Hunsur Road"),
    ("India", "Telangana",   "Hyderabad",   9, "Hitech City / Jubilee Hills"),
    ("India", "Tamil Nadu",  "Chennai",    10, "Anna Salai / OMR"),
    ("India", "Tamil Nadu",  "Coimbatore",  1, "Avinashi Road"),
    ("India", "West Bengal", "Kolkata",     2, "EM Bypass / VIP Road"),
    ("India", "Gujarat",     "Ahmedabad",   3, "SG Highway"),
    ("India", "Gujarat",     "Surat",       4, "Ring Road"),
    ("India", "Rajasthan",   "Jaipur",      5, "Ajmer Road"),
    ("India", "Uttar Pradesh","Lucknow",    6, "Hazratganj / Gomti Nagar"),
    ("India", "Uttar Pradesh","Kanpur",     7, "GT Road"),
    ("India", "Uttar Pradesh","Agra",       8, "Yamuna Expressway"),
    ("India", "Punjab",      "Chandigarh",  9, "Sector 17 / IT Park"),
    ("India", "Punjab",      "Ludhiana",   10, "Ferozepur Road"),
    ("India", "Madhya Pradesh","Bhopal",    1, "Hoshangabad Road"),
    ("India", "Madhya Pradesh","Indore",    2, "AB Road / Ring Road"),
    ("India", "Bihar",       "Patna",       3, "Bailey Road"),
    ("India", "Odisha",      "Bhubaneswar", 4, "NH-16 Corridor"),
    ("India", "Kerala",      "Thiruvananthapuram", 5, "NH-66"),
    ("India", "Kerala",      "Kochi",       6, "NH-544 / Edapally"),
    ("India", "Assam",       "Guwahati",    7, "GS Road"),
    ("India", "Jharkhand",   "Ranchi",      8, "Kanke Road"),
    ("India", "Chhattisgarh","Raipur",      9, "National Highway 53"),
    ("India", "Goa",         "Panaji",     10, "NH-66 Coastal"),
    # ── USA ──
    ("USA", "California",   "Los Angeles",    1, "I-405 / Hollywood Freeway"),
    ("USA", "California",   "San Francisco",  2, "Bay Bridge / 101"),
    ("USA", "California",   "San Diego",      3, "I-8 / Cabrillo Freeway"),
    ("USA", "New York",     "New York City",  4, "I-278 / FDR Drive"),
    ("USA", "New York",     "Buffalo",        5, "I-90 Thruway"),
    ("USA", "Texas",        "Houston",        6, "I-610 Loop"),
    ("USA", "Texas",        "Dallas",         7, "I-35E / LBJ Freeway"),
    ("USA", "Texas",        "Austin",         8, "MoPac Expressway"),
    ("USA", "Florida",      "Miami",          9, "I-95 / Brickell"),
    ("USA", "Florida",      "Orlando",       10, "I-4 Corridor"),
    ("USA", "Illinois",     "Chicago",        1, "I-90 / Lake Shore Drive"),
    ("USA", "Washington",   "Seattle",        2, "I-5 / SR-520"),
    ("USA", "Nevada",       "Las Vegas",      3, "Las Vegas Blvd / I-15"),
    ("USA", "Arizona",      "Phoenix",        4, "I-10 / Loop 101"),
    ("USA", "Georgia",      "Atlanta",        5, "I-285 / GA-400"),
    ("USA", "Massachusetts","Boston",         6, "I-93 / Mass Pike"),
    ("USA", "Pennsylvania", "Philadelphia",   7, "I-76 / Schuylkill"),
    ("USA", "Michigan",     "Detroit",        8, "I-75 / Lodge Freeway"),
    ("USA", "Colorado",     "Denver",         9, "I-25 / E-470"),
    ("USA", "Oregon",       "Portland",      10, "I-84 / Burnside Bridge"),
    # ── UK ──
    ("UK", "England",       "London",         1, "M25 / A406 North Circular"),
    ("UK", "England",       "Manchester",     2, "M60 Orbital / A57"),
    ("UK", "England",       "Birmingham",     3, "M6 / Spaghetti Junction"),
    ("UK", "England",       "Leeds",          4, "M1 / A64"),
    ("UK", "England",       "Liverpool",      5, "M62 / A5058"),
    ("UK", "Scotland",      "Glasgow",        6, "M8 / Kingston Bridge"),
    ("UK", "Scotland",      "Edinburgh",      7, "A720 City Bypass"),
    ("UK", "Wales",         "Cardiff",        8, "M4 / A48"),
    ("UK", "England",       "Bristol",        9, "M32 / Clifton Bridge"),
    ("UK", "England",       "Sheffield",     10, "M1 / Parkway"),
    # ── GERMANY ──
    ("Germany", "Bavaria",          "Munich",      1, "A8 / Mittlerer Ring"),
    ("Germany", "Berlin",           "Berlin",      2, "A100 / Stadtautobahn"),
    ("Germany", "Hamburg",          "Hamburg",     3, "A7 / Reeperbahn"),
    ("Germany", "North Rhine-Westphalia","Cologne", 4, "A3 / Leverkusener Brücke"),
    ("Germany", "Hesse",            "Frankfurt",   5, "A3 / A5 Interchange"),
    ("Germany", "Baden-Württemberg","Stuttgart",   6, "A8 / B14"),
    ("Germany", "Saxony",           "Dresden",     7, "A4 / B170"),
    ("Germany", "Lower Saxony",     "Hanover",     8, "A2 / Messeschnellweg"),
    ("Germany", "North Rhine-Westphalia","Dusseldorf", 9, "A46 / Rheinkniebrücke"),
    ("Germany", "Bavaria",          "Nuremberg",  10, "A9 / Frankenschnellweg"),
    # ── FRANCE ──
    ("France", "Île-de-France",     "Paris",       1, "Périphérique / A1"),
    ("France", "Auvergne-Rhône",    "Lyon",        2, "A6 / Périphérique Nord"),
    ("France", "Provence",          "Marseille",   3, "A7 / Bd Michelet"),
    ("France", "Occitanie",         "Toulouse",    4, "A620 Périphérique"),
    ("France", "Nouvelle-Aquitaine","Bordeaux",    5, "A630 / Pont d'Aquitaine"),
    ("France", "Grand Est",         "Strasbourg",  6, "A35 / A352"),
    # ── CHINA ──
    ("China", "Beijing",    "Beijing",       1, "3rd Ring Road / Chang'an Ave"),
    ("China", "Shanghai",   "Shanghai",      2, "Inner Ring Road / A20"),
    ("China", "Guangdong",  "Guangzhou",     3, "Inner Ring Expressway"),
    ("China", "Guangdong",  "Shenzhen",      4, "Shennan Avenue / G4E"),
    ("China", "Chongqing",  "Chongqing",     5, "Inner Ring / G75"),
    ("China", "Sichuan",    "Chengdu",       6, "2nd Ring Road"),
    ("China", "Hubei",      "Wuhan",         7, "2nd Ring Road / G42"),
    ("China", "Zhejiang",   "Hangzhou",      8, "Yan'an Road / G25"),
    ("China", "Jiangsu",    "Nanjing",       9, "Nanjing Ring Road"),
    ("China", "Liaoning",   "Shenyang",     10, "2nd Ring Road / G1"),
    ("China", "Shaanxi",    "Xi'an",         1, "South 2nd Ring Road"),
    ("China", "Shandong",   "Jinan",         2, "Jiwei Road / G2"),
    ("China", "Shandong",   "Qingdao",       3, "Haier Road / G20"),
    ("China", "Fujian",     "Xiamen",        4, "Xiamen Island Expressway"),
    ("China", "Tianjin",    "Tianjin",       5, "Zhonghuan Express / G2"),
    # ── JAPAN ──
    ("Japan", "Tokyo",      "Tokyo",         1, "Metropolitan Expressway C2"),
    ("Japan", "Osaka",      "Osaka",         2, "Hanshin Expressway / Route 16"),
    ("Japan", "Aichi",      "Nagoya",        3, "Meishin / Nagoya Ring Road"),
    ("Japan", "Hokkaido",   "Sapporo",       4, "Hokkaido Expressway E5"),
    ("Japan", "Miyagi",     "Sendai",        5, "Sendai Ring Road"),
    ("Japan", "Fukuoka",    "Fukuoka",       6, "Fukuoka Urban Expressway"),
    ("Japan", "Hiroshima",  "Hiroshima",     7, "San'yo Expressway"),
    ("Japan", "Kanagawa",   "Yokohama",      8, "Bay Shore Route / K3"),
    ("Japan", "Kyoto",      "Kyoto",         9, "Kyoto Bypass / Route 1"),
    ("Japan", "Okinawa",    "Naha",         10, "Okinawa Expressway"),
    # ── AUSTRALIA ──
    ("Australia", "New South Wales","Sydney",      1, "M1 / Harbour Bridge"),
    ("Australia", "Victoria",       "Melbourne",   2, "M1 / CityLink"),
    ("Australia", "Queensland",     "Brisbane",    3, "M3 / Pacific Motorway"),
    ("Australia", "Western Australia","Perth",      4, "Kwinana Freeway / M1"),
    ("Australia", "South Australia","Adelaide",    5, "South Eastern Freeway"),
    ("Australia", "Australian Capital Territory","Canberra", 6, "Tuggeranong Pkwy"),
    ("Australia", "Northern Territory","Darwin",   7, "Stuart Highway"),
    ("Australia", "Tasmania",       "Hobart",      8, "Southern Outlet / A6"),
    # ── CANADA ──
    ("Canada", "Ontario",       "Toronto",     1, "Highway 401 / Gardiner"),
    ("Canada", "Ontario",       "Ottawa",      2, "Highway 417 / Queensway"),
    ("Canada", "Quebec",        "Montreal",    3, "Autoroute 40 / Décarie"),
    ("Canada", "British Columbia","Vancouver",  4, "Trans-Canada Hwy / Granville"),
    ("Canada", "Alberta",       "Calgary",     5, "Deerfoot Trail / Ring Road"),
    ("Canada", "Alberta",       "Edmonton",    6, "Anthony Henday Drive"),
    ("Canada", "Manitoba",      "Winnipeg",    7, "Perimeter Highway"),
    ("Canada", "Nova Scotia",   "Halifax",     8, "Highway 102 / Bayers Rd"),
    # ── BRAZIL ──
    ("Brazil", "São Paulo",     "São Paulo",   1, "Marginal Pinheiros / Rodoanel"),
    ("Brazil", "Rio de Janeiro","Rio de Janeiro", 2, "Via Dutra / Linha Amarela"),
    ("Brazil", "Minas Gerais",  "Belo Horizonte", 3, "Anel Rodoviário"),
    ("Brazil", "Bahia",         "Salvador",    4, "BR-324 / Via Expressa"),
    ("Brazil", "Ceará",         "Fortaleza",   5, "BR-116 / Via Expressa"),
    ("Brazil", "Paraná",        "Curitiba",    6, "BR-277 / Linha Verde"),
    ("Brazil", "Pernambuco",    "Recife",      7, "BR-101 / Via Mangue"),
    ("Brazil", "Amazonas",      "Manaus",      8, "AM-010 / Via Expressa"),
    # ── RUSSIA ──
    ("Russia", "Moscow Oblast", "Moscow",      1, "MKAD / TTK Ring Road"),
    ("Russia", "Saint Petersburg","Saint Petersburg", 2, "KAD Ring Road"),
    ("Russia", "Novosibirsk",   "Novosibirsk", 3, "M-52 / Berdsk Highway"),
    ("Russia", "Yekaterinburg", "Yekaterinburg", 4, "EKAD / Siberian Tract"),
    ("Russia", "Tatarstan",     "Kazan",       5, "Inner Ring Road"),
    ("Russia", "Krasnodar",     "Krasnodar",   6, "Platovskiy / KAD"),
    # ── SOUTH AFRICA ──
    ("South Africa", "Gauteng",      "Johannesburg", 1, "N1 / N3 Interchange"),
    ("South Africa", "Gauteng",      "Pretoria",     2, "N1 / N4 Corridor"),
    ("South Africa", "Western Cape", "Cape Town",    3, "N2 / N1 / De Waal"),
    ("South Africa", "KwaZulu-Natal","Durban",       4, "N3 / N2 King Shaka"),
    ("South Africa", "Eastern Cape", "Port Elizabeth",5,"N2 / R10"),
    # ── NIGERIA ──
    ("Nigeria", "Lagos",    "Lagos",      1, "Lagos-Ibadan Expressway / 3rd Mainland"),
    ("Nigeria", "Abuja FCT","Abuja",      2, "Airport Road / Nnamdi Azikiwe"),
    ("Nigeria", "Kano",     "Kano",       3, "Kano Ring Road"),
    ("Nigeria", "Rivers",   "Port Harcourt", 4, "East-West Road"),
    ("Nigeria", "Oyo",      "Ibadan",     5, "Lagos-Ibadan Expressway"),
    # ── UAE ──
    ("UAE", "Dubai",        "Dubai",      1, "Sheikh Zayed Road / E11"),
    ("UAE", "Abu Dhabi",    "Abu Dhabi",  2, "Sheikh Maktoum Road / E10"),
    ("UAE", "Sharjah",      "Sharjah",    3, "University City Road / E311"),
    # ── SAUDI ARABIA ──
    ("Saudi Arabia", "Riyadh",   "Riyadh",     1, "King Fahd Road / Ring Road"),
    ("Saudi Arabia", "Jeddah",   "Jeddah",     2, "King Abdulaziz Road / Corniche"),
    ("Saudi Arabia", "Makkah",   "Mecca",      3, "Al-Haram / Mina Road"),
    ("Saudi Arabia", "Al-Madinah","Medina",    4, "Hijra Road / Quba"),
    # ── EGYPT ──
    ("Egypt", "Cairo",      "Cairo",      1, "Ring Road / Corniche el-Nil"),
    ("Egypt", "Alexandria", "Alexandria", 2, "Alexandria Desert Road"),
    ("Egypt", "Giza",       "Giza",       3, "Cairo-Alexandria Desert Road"),
    # ── SOUTH KOREA ──
    ("South Korea", "Seoul",     "Seoul",     1, "Olympic Expressway / Gangnam"),
    ("South Korea", "Busan",     "Busan",     2, "Nakdong River Expressway"),
    ("South Korea", "Incheon",   "Incheon",   3, "Incheon Airport Expressway"),
    ("South Korea", "Daegu",     "Daegu",     4, "Daegu Ring Road"),
    ("South Korea", "Gwangju",   "Gwangju",   5, "Beltway / National Route 1"),
    # ── MEXICO ──
    ("Mexico", "Mexico City", "Mexico City",  1, "Periférico / Viaducto"),
    ("Mexico", "Jalisco",     "Guadalajara",  2, "Periférico / López Mateos"),
    ("Mexico", "Nuevo León",  "Monterrey",    3, "Autopista 85 / Periférico"),
    ("Mexico", "Puebla",      "Puebla",       4, "ARCO Norte / Blvd Atlixcáyotl"),
    ("Mexico", "Yucatán",     "Mérida",       5, "Periférico / Paseo de Montejo"),
    # ── ARGENTINA ──
    ("Argentina", "Buenos Aires","Buenos Aires", 1, "General Paz / Autopista 25 Mayo"),
    ("Argentina", "Córdoba",    "Córdoba",      2, "Av. Circunvalación"),
    ("Argentina", "Rosario",    "Rosario",      3, "Av. de Circunvalación"),
    # ── INDONESIA ──
    ("Indonesia", "Jakarta",    "Jakarta",      1, "Tol Dalam Kota / JORR"),
    ("Indonesia", "West Java",  "Bandung",      2, "Tol Padaleunyi"),
    ("Indonesia", "East Java",  "Surabaya",     3, "MERR / Tol Waru"),
    ("Indonesia", "Bali",       "Denpasar",     4, "By Pass Ngurah Rai"),
    ("Indonesia", "North Sumatra","Medan",       5, "Ring Road / Tol Belmera"),
    # ── PAKISTAN ──
    ("Pakistan", "Punjab",      "Lahore",       1, "Ring Road / MM Alam"),
    ("Pakistan", "Sindh",       "Karachi",      2, "Northern Bypass / Shahrah-e-Faisal"),
    ("Pakistan", "Islamabad Capital","Islamabad",3, "Islamabad Highway / Expressway"),
    ("Pakistan", "KPK",         "Peshawar",     4, "GT Road / Ring Road"),
    # ── BANGLADESH ──
    ("Bangladesh", "Dhaka",     "Dhaka",        1, "Dhaka Bypass / Mirpur Road"),
    ("Bangladesh", "Chittagong","Chittagong",   2, "Chittagong Port Road"),
    # ── SRI LANKA ──
    ("Sri Lanka", "Western Province","Colombo",  1, "E01 / Baseline Road"),
    # ── NEPAL ──
    ("Nepal", "Bagmati",        "Kathmandu",    1, "Ring Road / Araniko Hwy"),
    # ── IRAN ──
    ("Iran", "Tehran",          "Tehran",       1, "Chamran Expressway / Ring Road"),
    ("Iran", "Isfahan",         "Isfahan",      2, "Nahjol-Balaghe Blvd"),
    ("Iran", "Mashhad",         "Mashhad",      3, "Vakil Abad Blvd"),
    # ── TURKEY ──
    ("Turkey", "Istanbul",      "Istanbul",     1, "O-1 / FSM Bridge"),
    ("Turkey", "Ankara",        "Ankara",       2, "Konya Road / O-4"),
    ("Turkey", "Izmir",         "Izmir",        3, "İzmir Ring Road / O-32"),
    # ── ITALY ──
    ("Italy", "Lazio",          "Rome",         1, "GRA / Via Appia"),
    ("Italy", "Lombardy",       "Milan",        2, "A51 Tangenziale Est"),
    ("Italy", "Campania",       "Naples",       3, "Raccordo Campano / A56"),
    ("Italy", "Veneto",         "Venice",       4, "A4 / Mestre Bypass"),
    ("Italy", "Tuscany",        "Florence",     5, "A1 / Firenze Sud"),
    # ── SPAIN ──
    ("Spain", "Madrid",         "Madrid",       1, "M-30 / A-2"),
    ("Spain", "Catalonia",      "Barcelona",    2, "B-23 / Ronda de Dalt"),
    ("Spain", "Valencia",       "Valencia",     3, "V-30 / A-3"),
    ("Spain", "Andalusia",      "Seville",      4, "SE-30 / A-49"),
    # ── PORTUGAL ──
    ("Portugal", "Lisbon",      "Lisbon",       1, "CRIL / A2"),
    ("Portugal", "Porto",       "Porto",        2, "Via de Cintura Interna"),
    # ── NETHERLANDS ──
    ("Netherlands", "North Holland","Amsterdam", 1, "A10 Ring Road"),
    ("Netherlands", "South Holland","Rotterdam", 2, "A20 / Maas Tunnel"),
    ("Netherlands", "South Holland","The Hague", 3, "A12 / A13"),
    # ── BELGIUM ──
    ("Belgium", "Brussels Capital","Brussels",  1, "R0 Brussels Ring / E19"),
    ("Belgium", "Flemish Region", "Antwerp",    2, "R1 Ring Road"),
    # ── SWITZERLAND ──
    ("Switzerland", "Zurich",   "Zurich",       1, "A1 / A3 / Cityring"),
    ("Switzerland", "Geneva",   "Geneva",       2, "A1 / Boulevard James-Fazy"),
    # ── AUSTRIA ──
    ("Austria", "Vienna",       "Vienna",       1, "A23 / Gürtel Ring"),
    # ── POLAND ──
    ("Poland", "Masovian",      "Warsaw",       1, "S2 / S7 Southern Bypass"),
    ("Poland", "Lesser Poland", "Krakow",       2, "A4 / Krakowska"),
    # ── SWEDEN ──
    ("Sweden", "Stockholm",     "Stockholm",    1, "Essingeleden / E4"),
    ("Sweden", "Västra Götaland","Gothenburg",  2, "E6 / E20"),
    # ── NORWAY ──
    ("Norway", "Oslo",          "Oslo",         1, "E18 / Ring 3"),
    # ── DENMARK ──
    ("Denmark", "Capital Region","Copenhagen",  1, "Motorring 3 / E20"),
    # ── FINLAND ──
    ("Finland", "Uusimaa",      "Helsinki",     1, "Ring III / E18"),
    # ── CZECH REPUBLIC ──
    ("Czech Republic","Prague",  "Prague",      1, "D0 City Ring / D1"),
    # ── HUNGARY ──
    ("Hungary", "Budapest",     "Budapest",     1, "M0 Ring Road"),
    # ── ROMANIA ──
    ("Romania", "Ilfov",        "Bucharest",    1, "DN1 / A3"),
    # ── UKRAINE ──
    ("Ukraine", "Kyiv City",    "Kyiv",         1, "Zhytomyr Hwy / Boryspil Hwy"),
    # ── GREECE ──
    ("Greece", "Attica",        "Athens",       1, "A6 / Attiki Odos"),
    # ── ISRAEL ──
    ("Israel", "Tel Aviv District","Tel Aviv",  1, "Ayalon Highway / Route 1"),
    # ── SINGAPORE ──
    ("Singapore", "Singapore",  "Singapore",    1, "PIE / CTE / AYE"),
    # ── MALAYSIA ──
    ("Malaysia", "Federal Territory","Kuala Lumpur", 1, "SPRINT / LDP"),
    ("Malaysia", "Johor",       "Johor Bahru",  2, "E2 / E1 North-South"),
    # ── THAILAND ──
    ("Thailand", "Bangkok",     "Bangkok",      1, "Si Rat Expressway / Outer Ring"),
    ("Thailand", "Chiang Mai",  "Chiang Mai",   2, "Super Highway / Ring Road"),
    # ── VIETNAM ──
    ("Vietnam", "Hanoi",        "Hanoi",        1, "Ring Road 3 / National Route 1"),
    ("Vietnam", "Ho Chi Minh",  "Ho Chi Minh City", 2, "Hanoi Highway / Ring Road 2"),
    # ── PHILIPPINES ──
    ("Philippines", "Metro Manila","Manila",    1, "EDSA / C5 Road"),
    ("Philippines", "Cebu",     "Cebu City",    2, "Mandaue-Mactan / MCTEP"),
    # ── NEW ZEALAND ──
    ("New Zealand", "Auckland", "Auckland",     1, "SH1 / Northern Motorway"),
    ("New Zealand", "Wellington","Wellington",  2, "Mt Victoria Tunnel / SH1"),
    # ── KENYA ──
    ("Kenya", "Nairobi",        "Nairobi",      1, "Thika Superhighway / Mombasa Rd"),
    ("Kenya", "Mombasa",        "Mombasa",      2, "Mombasa-Nairobi Road"),
    # ── ETHIOPIA ──
    ("Ethiopia", "Addis Ababa", "Addis Ababa",  1, "Ring Road / Bole Road"),
    # ── GHANA ──
    ("Ghana", "Greater Accra",  "Accra",        1, "N1 / Ring Road Central"),
    # ── TANZANIA ──
    ("Tanzania", "Dar es Salaam","Dar es Salaam",1,"Morogoro Road / Nelson Mandela"),
    # ── MOROCCO ──
    ("Morocco", "Casablanca-Settat","Casablanca",1,"A1 / Bd Mohammed VI"),
    ("Morocco", "Rabat-Salé",   "Rabat",        2,"A1 / Av Mohammed VI"),
    # ── ALGERIA ──
    ("Algeria", "Algiers",      "Algiers",      1,"East-West Highway / Ring Road"),
    # ── CHILE ──
    ("Chile", "Santiago Metro", "Santiago",     1,"Autopista Central / Costanera Norte"),
    # ── COLOMBIA ──
    ("Colombia", "Bogotá D.C.", "Bogotá",       1,"Calle 80 / Autopista Sur"),
    ("Colombia", "Antioquia",   "Medellín",     2,"Autopista Norte / Periférico"),
    # ── PERU ──
    ("Peru", "Lima Province",   "Lima",         1,"Panamericana Sur / Javier Prado"),
    # ── VENEZUELA ──
    ("Venezuela", "Capital District","Caracas",  1,"Autopista Francisco Fajardo"),
    # ── CUBA ──
    ("Cuba", "Havana",          "Havana",       1,"Autopista Nacional / Malecón"),
]

# Build lookup structures
LOCATION_INDEX = []   # list of dicts for fast search
for country, state, city, seg, road in WORLD_LOCATIONS:
    LOCATION_INDEX.append({
        "country": country,
        "state":   state,
        "city":    city,
        "segment": seg,
        "road":    road,
        "label":   f"{city}, {state}, {country}",
        "search_key": f"{country} {state} {city}".lower(),
    })

ALL_COUNTRIES = sorted(set(e["country"] for e in LOCATION_INDEX))

def search_locations(query: str):
    """Return matching location entries for a query string."""
    q = query.strip().lower()
    if not q:
        return LOCATION_INDEX
    return [e for e in LOCATION_INDEX if q in e["search_key"]]

# Legacy segment label (used for chart titles)
SEGMENT_LABEL = {
    1: "Segment 1", 2: "Segment 2", 3: "Segment 3", 4: "Segment 4",
    5: "Segment 5", 6: "Segment 6", 7: "Segment 7", 8: "Segment 8",
    9: "Segment 9", 10: "Segment 10",
}

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

# Show all locations directly grouped by country in the selectbox
loc_options = [
    f"{e['city']} — {e['state']}, {e['country']}"
    for e in LOCATION_INDEX
]
matched_locs = LOCATION_INDEX

selected_option = st.sidebar.selectbox(
    "📍 Select Location",
    options=loc_options,
)

# Resolve back to location entry
sel_idx = loc_options.index(selected_option)
selected_loc = matched_locs[sel_idx]
segment_id = selected_loc["segment"]

st.sidebar.success(
    f"📍 **{selected_loc['city']}**  \n"
    f"{selected_loc['state']}, {selected_loc['country']}  \n"
    f"🛣️ {selected_loc['road']}  \n"
    f"📊 Segment {segment_id}"
)

st.sidebar.markdown("---")
forecast_horizon = st.sidebar.slider("⏱️ Forecast Horizon (hours)", min_value=1, max_value=4, value=2)
model_choice = st.sidebar.radio("🤖 Prediction Model", ["Random Forest", "XGBoost"])
show_heatmap = st.sidebar.checkbox("🗺️ Show Congestion Heatmap", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "Predictions represent vehicle volume counts for the selected segment "
    "1–4 hours into the future based on recent traffic patterns."
)

# Logout button at the bottom of the sidebar
st.sidebar.markdown("---")
if st.sidebar.button("🔒 Logout", use_container_width=True):
    _do_logout()
    st.rerun()
st.sidebar.caption(f"Signed in as **{st.session_state.login_user}**")

# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────

# ── Branded top header bar ──
st.markdown(
    f"""
    <div class="top-bar">
      <div class="top-bar-left">
        <span class="top-bar-logo">🚦</span>
        <div>
          <p class="top-bar-title">Smart Urban Traffic Forecasting</p>
          <p class="top-bar-sub">SDG 11 — Sustainable Cities &amp; Communities</p>
        </div>
      </div>
      <span class="top-bar-badge">Live Dashboard</span>
    </div>
    """,
    unsafe_allow_html=True,
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

location_label = f"{selected_loc['city']}, {selected_loc['state']}, {selected_loc['country']}"
road_label = selected_loc['road']
st.markdown(
    f"### 📊 Current Conditions — 📍 {selected_loc['city']}"
    f"&nbsp;&nbsp;<span style='color:#57606a;font-size:0.9rem;'>({selected_loc['state']}, {selected_loc['country']})</span>",
    unsafe_allow_html=True,
)
st.caption(f"🛣️ Key Road/District: **{road_label}** &nbsp;|&nbsp; Traffic data: Segment {segment_id}")
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
    plt.style.use("dark_background")
    fig_f, ax_f = plt.subplots(figsize=(8, 3), facecolor="#161b22")
    ax_f.set_facecolor("#0f1117")
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
    ax_f.axhline(current_vol, color="#58a6ff", linestyle="--", linewidth=1.2, label="Current volume")
    ax_f.set_ylabel("Predicted Vehicle Volume", color="#8b949e")
    ax_f.set_title(f"Forecast — Segment {segment_id} ({model_choice})", color="#e6edf3")
    ax_f.tick_params(colors="#8b949e")
    ax_f.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    ax_f.grid(axis="y", linestyle="--", alpha=0.25, color="#30363d")
    for spine in ax_f.spines.values():
        spine.set_edgecolor("#30363d")
    fig_f.tight_layout()
    st.pyplot(fig_f, use_container_width=True)
    plt.close(fig_f)

st.markdown("---")

# ── HISTORICAL TREND ──
st.markdown("### 📈 Historical Traffic Volume (Last 7 Days)")
last_7d = df_seg[df_seg["timestamp"] >= df_seg["timestamp"].max() - pd.Timedelta(days=7)]

fig_h, ax_h = plt.subplots(figsize=(12, 3.5), facecolor="#161b22")
ax_h.set_facecolor("#0f1117")
ax_h.plot(last_7d["timestamp"], last_7d["vehicle_volume"], color="#58a6ff", linewidth=1.4, alpha=0.9)
ax_h.fill_between(last_7d["timestamp"], last_7d["vehicle_volume"], alpha=0.12, color="#58a6ff")
ax_h.set_ylabel("Vehicle Volume (veh/hr)", color="#8b949e")
ax_h.set_title(f"Segment {segment_id} — Last 7 Days", color="#e6edf3")
ax_h.tick_params(colors="#8b949e")
ax_h.grid(True, linestyle="--", alpha=0.2, color="#30363d")
for spine in ax_h.spines.values():
    spine.set_edgecolor("#30363d")
fig_h.autofmt_xdate()
fig_h.tight_layout()
st.pyplot(fig_h, use_container_width=True)
plt.close(fig_h)

st.markdown("---")

# ── PEAK HOUR ANALYSIS ──
st.markdown("### 🕐 Average Volume by Hour of Day")
hourly_avg = df_seg.groupby("hour")["vehicle_volume"].mean().reset_index()

fig_p, ax_p = plt.subplots(figsize=(10, 3), facecolor="#161b22")
ax_p.set_facecolor("#0f1117")
bars = ax_p.bar(hourly_avg["hour"], hourly_avg["vehicle_volume"], edgecolor="none", width=0.7)
# Colour peaks
for bar, h in zip(bars, hourly_avg["hour"]):
    if h in [7, 8, 9, 17, 18, 19]:
        bar.set_color("#e74c3c")
    else:
        bar.set_color("#1f6feb")

ax_p.set_xlabel("Hour of Day", color="#8b949e")
ax_p.set_ylabel("Avg Vehicle Volume", color="#8b949e")
ax_p.set_title(f"Segment {segment_id} — Hourly Traffic Profile (red = peak hours)", color="#e6edf3")
ax_p.set_xticks(range(24))
ax_p.tick_params(colors="#8b949e")
ax_p.grid(axis="y", linestyle="--", alpha=0.2, color="#30363d")
for spine in ax_p.spines.values():
    spine.set_edgecolor("#30363d")
fig_p.tight_layout()
st.pyplot(fig_p, use_container_width=True)
plt.close(fig_p)

st.markdown("---")

# ── CONGESTION HEATMAP ──
if show_heatmap:
    st.markdown("### 🗺️ Network-Wide Congestion Heatmap")
    try:
        from evaluation import generate_folium_heatmap
        latest_snapshot = (
            df.sort_values("timestamp")
            .groupby("segment_id")
            .last()
            .reset_index()[["segment_id", "vehicle_volume"]]
        )
        map_path = os.path.join(_ROOT, "assets", "traffic_heatmap.html")
        os.makedirs(os.path.join(_ROOT, "assets"), exist_ok=True)
        generate_folium_heatmap(latest_snapshot, output_path=map_path)
        with open(map_path, "r", encoding="utf-8") as f:
            map_html = f.read()
        st.components.v1.html(map_html, height=450)
    except ImportError:
        st.info("Install `folium` to enable the interactive heatmap: `pip install folium`")
    except Exception as e:
        st.warning(f"Heatmap could not be generated: {e}")
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
