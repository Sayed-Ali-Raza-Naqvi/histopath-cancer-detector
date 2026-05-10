import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(__file__))
from config import WEIGHTS_PATH, DEVICE, model_store
from gradcam import load_model
from routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading model from: {WEIGHTS_PATH}")
    print(f"Device: {DEVICE}")
    model_store["model"] = load_model(WEIGHTS_PATH, DEVICE)
    print("Model loaded and ready.")
    yield
    
    model_store.clear()
    print("Model unloaded.")


app = FastAPI(
    title = "Histopathology Cancer Detector",
    description = "Explainable AI for breast cancer detection using EfficientNet-B0 and Grad-CAM.",
    version = "1.0.0",
    lifespan = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

app.include_router(router, prefix="/api/v1")