import os
import io
import json
import base64
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException

from schemas import PredictionResponse, HealthResponse, ModelInfoResponse
from gradcam import run_inference_with_gradcam
from config import BASE_DIR, DEVICE, THRESHOLD, model_store, transform

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_FILE_SIZE_MB = 10
MIN_IMAGE_SIZE = 32
MAX_IMAGE_SIZE = 2048


def validate_image_file(file: UploadFile, contents: bytes):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code = 415,
            detail = f"Unsupported file type: {file.content_type}. Allowed: JPEG, PNG."
        )
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code = 413,
            detail = f"File too large: {size_mb:.1f}MB. Maximum: {MAX_FILE_SIZE_MB}MB."
        )


def validate_image_dimensions(image: Image.Image):
    w, h = image.size
    if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
        raise HTTPException(
            status_code = 422,
            detail = f"Image too small: {w}x{h}px. Minimum: {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}px."
        )
    if w > MAX_IMAGE_SIZE or h > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code = 422,
            detail = f"Image too large: {w}x{h}px. Maximum: {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE}px."
        )


def numpy_to_base64(image_array: np.ndarray) -> str:
    pil_image = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


@router.get("/")
def root():
    return {"message": "Histopathology Cancer Detector API is running. Visit /docs for the API documentation."}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status  = "healthy",
        model_loaded = "model" in model_store,
        device = str(DEVICE),
        threshold = THRESHOLD
    )


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    metrics_path = os.path.join(BASE_DIR, "model", "weights", "test_metrics.json")

    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code = 404,
            detail = "test_metrics.json not found. Run model/test.py first."
        )

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    optimal = metrics.get("optimal", {})

    return ModelInfoResponse(
        architecture = "EfficientNet-B0 (fine-tuned)",
        dataset = "PatchCamelyon (PCam) — 327,680 histopathology patches",
        threshold = optimal.get("threshold",  0.0),
        test_auc_roc = optimal.get("auc_roc",    0.0),
        test_accuracy = optimal.get("accuracy",   0.0),
        test_f1 = optimal.get("f1",         0.0),
        test_precision = optimal.get("precision",  0.0),
        test_recall = optimal.get("recall",     0.0),
        disclaimer = "For research use only. Not a clinical diagnostic tool."
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    validate_image_file(file, contents)

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code = 422,
            detail = "Could not decode image. Ensure the file is a valid JPEG or PNG."
        )

    validate_image_dimensions(image)

    if "model" not in model_store:
        raise HTTPException(
            status_code = 503,
            detail = "Model not loaded. Please try again later."
        )

    try:
        result = run_inference_with_gradcam(
            pil_image = image,
            model = model_store["model"],
            transform = transform,
            device = DEVICE,
            threshold = THRESHOLD
        )
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"Inference failed: {str(e)}"
        )

    return PredictionResponse(
        prediction = result["prediction"],
        tumor_probability = result["tumor_probability"],
        confidence = result["confidence"],
        threshold_used = THRESHOLD,
        gradcam_image = numpy_to_base64(result["heatmap"]),
    )