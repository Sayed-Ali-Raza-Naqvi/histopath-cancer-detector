from pydantic import BaseModel


class PredictionResponse(BaseModel):
    prediction: str
    tumor_probability: float
    confidence: float
    threshold_used: float
    gradcam_image: str
    model_version: str = "1.0.0"
    disclaimer: str = "For research use only. Not a clinical diagnostic tool."


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    threshold: float


class ModelInfoResponse(BaseModel):
    architecture: str
    dataset: str
    threshold: float
    test_auc_roc: float
    test_accuracy: float
    test_f1: float
    test_precision: float
    test_recall: float
    disclaimer: str