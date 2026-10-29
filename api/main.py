import os

import mlflow
import pandas as pd

from fastapi import FastAPI, HTTPException

from api.schemas import PredictionRequest, PredictionResponse


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "hydraulic-valve-classifier",
)

MODEL_ALIAS = os.getenv(
    "MODEL_ALIAS",
    "champion",
)


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


app = FastAPI(
    title="Hydraulic System Health API",
    description="Predicts hydraulic valve condition using a registered ML model.",
    version="1.0.0",
)


def load_model():
    """Load the current model assigned to the MLflow alias."""
    return mlflow.pyfunc.load_model(MODEL_URI)


model = load_model()


CONDITION_DESCRIPTIONS = {
    73: "Close to total failure",
    80: "Severe lag",
    90: "Small lag",
    100: "Optimal switching behavior",
}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/model")
def model_info():
    return {
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "target": "valve_condition",
        "features": 102,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    if len(request.features) != 102:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 102 features, received {len(request.features)}",
        )

    try:
        input_data = pd.DataFrame([request.features])

        prediction = int(model.predict(input_data)[0])

        return PredictionResponse(
            valve_condition=prediction,
            condition_description=CONDITION_DESCRIPTIONS[prediction],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )