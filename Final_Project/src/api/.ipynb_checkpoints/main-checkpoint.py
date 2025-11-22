from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import datetime
from pathlib import Path
import uuid
import joblib
import pandas as pd
import sys

# --- Paths ---
PROJECT_DIR = Path(__file__).resolve().parents[2]  # .../ML_ops/Final_Project
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Import wrapper so unpickling can find PreprocessAndAutoML
from models.wrapper import PreprocessAndAutoML  # noqa: F401

MODEL_PATH = PROJECT_DIR / "models" / "model.pkl"
MONITORING_DIR = PROJECT_DIR / "monitoring"
LOG_PATH = MONITORING_DIR / "predictions_log.csv"

MONITORING_DIR.mkdir(parents=True, exist_ok=True)

# --- Load trained model (wrapper: preprocessor + AutoML) ---
model = joblib.load(MODEL_PATH)

# --- FastAPI app ---
app = FastAPI(title="Car Price Prediction API")


# --- Request/response schemas ---
class CarInput(BaseModel):
    brand: str
    year: int
    engine_size: float
    fuel_type: str
    transmission: str
    mileage: int
    condition: str
    model: str


class PredictionResponse(BaseModel):
    request_id: str
    predictions: List[float]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(cars: List[CarInput]):
    # Convert request to DataFrame
    records = [c.dict() for c in cars]
    df = pd.DataFrame(records)

    # Match training column names
    df_model = df.rename(
        columns={
            "brand": "Brand",
            "year": "Year",
            "engine_size": "Engine Size",
            "fuel_type": "Fuel Type",
            "transmission": "Transmission",
            "mileage": "Mileage",
            "condition": "Condition",
            "model": "Model",
        }
    )

    # Predict
    preds = model.predict(df_model)
    preds_list = [float(p) for p in preds]

    # Simple monitoring log
    ts = datetime.utcnow().isoformat()
    request_id = str(uuid.uuid4())

    log_df = df_model.copy()
    log_df["prediction"] = preds_list
    log_df["timestamp"] = ts
    log_df["request_id"] = request_id

    header = not LOG_PATH.exists()
    log_df.to_csv(LOG_PATH, mode="a", header=header, index=False)

    return PredictionResponse(request_id=request_id, predictions=preds_list)