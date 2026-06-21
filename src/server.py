"""
Title: FastAPI backend for deepfake prediction
Author: Vlad Cozma
Creation Date: 24.05.2026
"""

from __future__ import annotations

from pathlib import Path
import io

import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
MODEL_PATH = PROJECT_ROOT / "models" / "deepfake_detector.h5"
WEB_DIR = APP_ROOT / "website"

IMG_SIZE = 224
PRED_THRESHOLD = 0.5

app = FastAPI(title="Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model():
    global _model
    
    _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def prepare_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    image_array = np.array(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Only PNG and JPEG images are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        image_tensor = prepare_image(contents)
        model = get_model()
        pred_prob = model.predict(image_tensor, verbose=0).ravel()[0]
        pred = int(pred_prob > PRED_THRESHOLD)
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    label = "REAL" if pred == 1 else "DEEPFAKE"
    confidence = float(pred_prob) if pred == 1 else float(1.0 - pred_prob)

    return {
        "label": label,
        "confidence": confidence,
    }


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
