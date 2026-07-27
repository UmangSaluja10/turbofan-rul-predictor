"""
FastAPI backend for the Turbofan RUL Predictor.

Loads the model/scaler/feature list saved by train_model.py and exposes
a REST API for the frontend to call.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs to see the auto-generated Swagger UI
(this doubles as your "API Documentation" deliverable).
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "rul_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

# ---------------------------------------------------------------------------
# Load model artifacts ONCE at startup (not per-request — important for speed)
# ---------------------------------------------------------------------------
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH)):
    raise RuntimeError(
        "Model artifacts not found. Run `python train_model.py` from the "
        "project root first to generate models/rul_model.pkl, scaler.pkl, "
        "and feature_columns.pkl."
    )

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)

RUL_CAP = 125  # must match the cap used in train_model.py

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Turbofan Engine RUL Predictor API",
    description="Predicts Remaining Useful Life (in cycles) of a jet engine from sensor readings.",
    version="1.0.0",
)

# Allow the frontend (served from a different origin, e.g. file:// or localhost:5500)
# to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class SensorReading(BaseModel):
    """
    One row of sensor + operating-setting readings for a single engine
    at a single point in time. Field names must match feature_columns.pkl.
    """
    op_setting_1: float = Field(..., example=-0.0007)
    op_setting_2: float = Field(..., example=0.0004)
    op_setting_3: float = Field(..., example=100.0)
    sensor_2: float = Field(..., example=642.5)
    sensor_3: float = Field(..., example=1590.0)
    sensor_4: float = Field(..., example=1400.0)
    sensor_7: float = Field(..., example=554.5)
    sensor_8: float = Field(..., example=2388.0)
    sensor_9: float = Field(..., example=9050.0)
    sensor_11: float = Field(..., example=47.3)
    sensor_12: float = Field(..., example=522.0)
    sensor_13: float = Field(..., example=2388.0)
    sensor_14: float = Field(..., example=8140.0)
    sensor_15: float = Field(..., example=8.4)
    sensor_17: float = Field(..., example=392.0)
    sensor_20: float = Field(..., example=39.0)
    sensor_21: float = Field(..., example=23.4)


class PredictionResponse(BaseModel):
    predicted_rul: float
    status: str
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Turbofan RUL Predictor API is running. Visit /docs for API documentation."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/feature-schema")
def feature_schema():
    """Lets the frontend know exactly which fields/order the model expects."""
    return {"features": feature_columns}


def classify_status(predicted_rul: float) -> tuple[str, str]:
    if predicted_rul <= 20:
        return "critical", "Engine is nearing end of life. Schedule maintenance immediately."
    elif predicted_rul <= 50:
        return "warning", "Engine shows signs of degradation. Plan maintenance soon."
    else:
        return "healthy", "Engine is operating within a healthy range."


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    try:
        data: Dict[str, float] = reading.dict()
        # Build the feature vector in the EXACT order the model was trained on
        ordered_values = [data[col] for col in feature_columns]
        X = np.array(ordered_values).reshape(1, -1)
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]
        pred = float(np.clip(pred, 0, RUL_CAP))

        status, message = classify_status(pred)

        return PredictionResponse(
            predicted_rul=round(pred, 1),
            status=status,
            message=message,
        )
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))