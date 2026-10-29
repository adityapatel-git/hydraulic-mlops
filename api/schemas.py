from pydantic import BaseModel


class PredictionRequest(BaseModel):
    features: dict[str, float]


class PredictionResponse(BaseModel):
    valve_condition: int
    condition_description: str