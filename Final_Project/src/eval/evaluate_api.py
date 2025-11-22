import os
import pandas as pd
import numpy as np
import requests

DATA_DIR = "Final_Project/data"
API_URL = "http://0.0.0.0:8000/predict"


def load_test_data():
    X_test = pd.read_parquet(os.path.join(DATA_DIR, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(DATA_DIR, "y_test.parquet"))["Price"]
    return X_test, y_test


def build_payload(df):
    """Convert X_test rows into the JSON the API expects (list of CarInput)."""
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "brand": row["Brand"],
                "year": int(row["Year"]),
                "engine_size": float(row["Engine Size"]),
                "fuel_type": row["Fuel Type"],
                "transmission": row["Transmission"],
                "mileage": int(row["Mileage"]),
                "condition": row["Condition"],
                "model": row["Model"],
            }
        )
    return records


def call_api(payload):
    """Send list[CarInput] to the /predict endpoint and return np.array of predictions."""
    resp = requests.post(API_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return np.array(data["predictions"], dtype=float)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    X_test, y_test = load_test_data()

    # Use a subset to keep it light
    X_sub = X_test.head(200).copy()
    y_sub = y_test.head(200).copy()

    # ---------- 1) Original test data ----------
    payload_orig = build_payload(X_sub)
    preds_orig = call_api(payload_orig)
    rmse_orig = rmse(y_sub.values, preds_orig)
    print(f"RMSE on original test subset: {rmse_orig:.2f}")

    # ---------- 2) Changed test data (≥2 features changed) ----------
    X_changed = X_sub.copy()
    # Make cars "worse": older and higher mileage
    X_changed["Mileage"] = (X_changed["Mileage"] * 1.5).astype(int)
    X_changed["Year"] = X_changed["Year"] - 1

    payload_changed = build_payload(X_changed)
    preds_changed = call_api(payload_changed)
    rmse_changed = rmse(y_sub.values, preds_changed)
    print(f"RMSE on changed test subset: {rmse_changed:.2f}")

    print(f"\nDelta RMSE (changed - original): {rmse_changed - rmse_orig:.2f}")

    print(
        "\nRequests have been logged to "
        "Final_Project/monitoring/predictions_log.csv for monitoring."
    )


if __name__ == "__main__":
    main()