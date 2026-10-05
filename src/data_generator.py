"""
Synthetic Traffic Dataset Generator
------------------------------------
Generates realistic spatial-temporal traffic data for model training and testing.
Simulates hourly vehicle volume, speed, and weather parameters across multiple road segments.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_traffic_dataset(
    n_segments: int = 10,
    days: int = 365,
    seed: int = 42,
    output_path: str = "data/raw/traffic_data.csv",
) -> pd.DataFrame:
    """
    Generate a synthetic hourly traffic dataset.

    Parameters
    ----------
    n_segments : int
        Number of road segments to simulate.
    days : int
        Number of days of historical data to generate.
    seed : int
        Random seed for reproducibility.
    output_path : str
        Path to save the generated CSV.

    Returns
    -------
    pd.DataFrame
        Generated traffic dataset.
    """
    np.random.seed(seed)

    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=h) for h in range(days * 24)]
    records = []

    for segment_id in range(1, n_segments + 1):
        base_capacity = np.random.randint(800, 2000)

        for ts in timestamps:
            hour = ts.hour
            day_of_week = ts.weekday()  # 0=Monday, 6=Sunday
            month = ts.month

            # ---- Temporal demand pattern ----
            # Morning peak: 7-9 AM, Evening peak: 5-7 PM
            if 7 <= hour <= 9:
                peak_factor = 1.8
            elif 17 <= hour <= 19:
                peak_factor = 1.9
            elif 0 <= hour <= 5:
                peak_factor = 0.3
            else:
                peak_factor = 1.0

            # Weekend reduction
            weekend_factor = 0.65 if day_of_week >= 5 else 1.0

            # Seasonal effect (summer busier in some cities)
            seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (month - 3) / 12)

            vehicle_volume = int(
                base_capacity
                * peak_factor
                * weekend_factor
                * seasonal_factor
                * np.random.uniform(0.85, 1.15)
            )
            vehicle_volume = max(0, vehicle_volume)

            # Speed inversely correlated with volume
            free_flow_speed = np.random.uniform(50, 80)
            congestion_ratio = min(vehicle_volume / base_capacity, 1.5)
            avg_speed = free_flow_speed * (1 - 0.6 * congestion_ratio) + np.random.normal(0, 2)
            avg_speed = max(5.0, round(avg_speed, 2))

            # Weather parameters
            temperature = 15 + 10 * np.sin(2 * np.pi * (month - 3) / 12) + np.random.normal(0, 3)
            rainfall_mm = max(0.0, np.random.exponential(0.5) if np.random.rand() < 0.2 else 0.0)
            visibility_km = max(0.5, 10.0 - rainfall_mm * 2 + np.random.normal(0, 0.5))

            # Congestion severity label
            if congestion_ratio < 0.6:
                severity = "Low"
            elif congestion_ratio < 1.0:
                severity = "Medium"
            else:
                severity = "High"

            records.append(
                {
                    "timestamp": ts,
                    "segment_id": segment_id,
                    "vehicle_volume": vehicle_volume,
                    "avg_speed_kmh": avg_speed,
                    "temperature_c": round(temperature, 2),
                    "rainfall_mm": round(rainfall_mm, 2),
                    "visibility_km": round(visibility_km, 2),
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "month": month,
                    "is_weekend": int(day_of_week >= 5),
                    "congestion_severity": severity,
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"[✓] Dataset generated: {len(df):,} rows → saved to '{output_path}'")
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data/raw", exist_ok=True)
    df = generate_traffic_dataset()
    print(df.head())
    print(df.dtypes)
