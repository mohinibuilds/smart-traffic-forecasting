# 🚦 Smart Urban Traffic Congestion & Resource Demand Forecasting System

> **SDG 11 — Sustainable Cities and Communities**  
> Providing municipal authorities with actionable, data-driven traffic control strategies to reduce congestion, vehicular idling times, and greenhouse gas emissions.

---

## 📁 Project Structure

```
smart_traffic_forecasting/
│
├── app.py                        # Streamlit interactive dashboard
├── requirements.txt              # Python dependencies
│
├── src/
│   ├── data_generator.py         # Synthetic spatial-temporal dataset generator
│   ├── preprocessing.py          # Data cleaning, cyclical encoding, lag features, scaling
│   ├── baseline_models.py        # Random Forest & XGBoost training + evaluation
│   ├── lstm_model.py             # TensorFlow LSTM sequential model
│   ├── evaluation.py             # Metrics, plots, Folium heatmap utilities
│   └── train_all.py              # End-to-end training orchestrator
│
├── data/
│   └── raw/
│       └── traffic_data.csv      # Generated after running train_all.py
│
├── models/
│   └── saved/
│       ├── feature_scaler.pkl    # Fitted MinMaxScaler
│       ├── random_forest.pkl     # Trained RF model
│       ├── xgboost.pkl           # Trained XGBoost model
│       └── lstm_final.keras      # Trained LSTM model
│
├── assets/
│   └── traffic_heatmap.html      # Folium-generated congestion heatmap
│
└── notebooks/                    # (Optional) Jupyter exploration notebooks
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.x |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn, folium |
| Baseline ML | scikit-learn (Random Forest), XGBoost |
| Deep Learning | TensorFlow / Keras (LSTM) |
| Time-Series | statsmodels (ARIMA-ready) |
| Dashboard | Streamlit |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all models (first time only — generates data + trains RF & XGBoost)
```bash
python src/train_all.py
```
> **Note:** Skip this step if `models/saved/` already contains `.pkl` files.

### 3. Launch the interactive dashboard
```bash
python -m streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

**Default login credentials:**
| Username | Password |
|----------|----------|
| admin    | traffic123 |
| demo     | demo |

> You can also create your own account from the **Create Account** tab on the login page.

---

## 🧠 Technical Workflow

### Data Acquisition & Preprocessing
- Simulates hourly vehicle volume counts, speed data, peak/non-peak time tags, and weather parameters across 10 road segments over 365 days.
- Missing values handled via **temporal interpolation** (`pandas`).
- Cyclical time features (`hour`, `day_of_week`, `month`) encoded with **sine/cosine transformations**.

### Feature Engineering
| Feature Type | Details |
|---|---|
| Lag features | t−1, t−2, t−3, t−24 (yesterday same hour) |
| Rolling averages | 3-h, 6-h, 24-h moving average & std deviation |
| Weather | Temperature, rainfall, visibility |
| Cyclical encoding | Hour sin/cos, Day-of-week sin/cos, Month sin/cos |

### Model Architecture

#### Baseline — XGBoost / Random Forest
- Tabular feature matrix → `XGBRegressor` / `RandomForestRegressor`
- Time-aware train/test split (no shuffle) to prevent data leakage
- Feature importance ranking provided

#### Advanced — LSTM Neural Network
```
Input → LSTM(128) → BatchNorm → Dropout(0.2)
      → LSTM(64)  → BatchNorm → Dropout(0.2)
      → Dense(32) → Dense(1) [output: vehicle volume]
```
- Input sequence: **last 24 hours** of engineered features  
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### Evaluation Metrics
| Metric | Description |
|--------|------------|
| MAE | Mean Absolute Error (vehicles/hr) |
| RMSE | Root Mean Squared Error |
| R² Score | Coefficient of determination |

---

## 📊 Dashboard Features

| Feature | Description |
|---------|------------|
| KPI Cards | Current volume, speed, rainfall, congestion severity |
| Forecast Cards | +1h to +4h ahead predicted volumes with colour-coded severity |
| Forecast Bar Chart | Visual comparison across forecast horizon |
| Historical Trend | Last 7-day volume time series |
| Hourly Profile | Average volume per hour (peak hours highlighted) |
| Congestion Heatmap | Interactive Folium map of all 10 segments |
| Re-routing Alerts | Automated Low / Medium / High severity recommendations |

---

## 🌍 SDG Impact

| SDG Target | Contribution |
|---|---|
| **11.2** | Improves urban transport planning with 2–4 hour advance congestion warnings |
| **11.6** | Reduces vehicle idling → lower localized greenhouse gas emissions |
| **11.b** | Supports data-driven policy for resilient and sustainable cities |

---

## 📋 License
MIT License — open for municipal and research use.
