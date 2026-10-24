"""
FastAPI Backend for Tomato Leaf Disease Classification
Serving YOLO11n-cls, Novel CNN Inception, and MobileViT Transformer.
"""

import io
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from app.model_engine import engine
from app.disease_knowledge import TOMATO_DISEASES

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = STATIC_DIR / "samples"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load all models on server startup for instantaneous inference."""
    print("[*] Starting Tomato Crop Disease AI Server...", flush=True)
    engine.load_models()
    yield
    print("[*] Shutting down Tomato Crop Disease AI Server...", flush=True)


app = FastAPI(
    title="Tomato Crop Disease Unified AI Dashboard",
    description="Multi-model classification API comparing YOLO11, Novel Inception CNN, and MobileViT Transformer with comprehensive agronomic remediation measures.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for seamless dashboard communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "models_loaded": engine.is_loaded,
        "supported_models": ["yolo", "novel_cnn", "mobilevit"]
    }


@app.get("/api/models")
async def get_models():
    """Retrieve metadata and architectural specifications for all 3 models."""
    return {
        "models": engine.get_models_metadata()
    }


@app.get("/api/diseases")
async def get_diseases():
    """Retrieve full catalog of tomato leaf diseases and farmer remediation advice."""
    return {
        "diseases": TOMATO_DISEASES
    }


@app.get("/api/samples")
async def get_sample_images():
    """Return available sample test leaf images."""
    samples = []
    if SAMPLES_DIR.exists():
        for f in sorted(os.listdir(SAMPLES_DIR)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                name = f.replace(".jpg", "").replace("sample_", "").replace("_", " ").title()
                samples.append({
                    "filename": f,
                    "title": name,
                    "url": f"/static/samples/{f}"
                })
    return {"samples": samples}


def _read_image(file_bytes: bytes) -> Image.Image:
    """Helper to safely decode and validate uploaded image."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify() # Verify file integrity
        # Re-open after verify()
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image file. Please upload a valid JPG/PNG image. Error: {str(e)}"
        )


@app.post("/api/predict")
async def predict_single_model(
    file: UploadFile = File(...),
    model_id: str = Form(default="yolo")
):
    """
    Run disease classification on the uploaded leaf image using the selected model.
    model_id options: 'yolo', 'novel_cnn', 'mobilevit'
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image (JPEG, PNG, WEBP)."
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    img = _read_image(file_bytes)

    try:
        result = engine.predict(model_id=model_id, img=img)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference processing failed: {str(e)}")


@app.post("/api/predict_all")
async def predict_all_models(
    file: UploadFile = File(...)
):
    """
    Run disease classification simultaneously across all 3 models (YOLO, Novel CNN, MobileViT)
    to generate side-by-side comparison and majority consensus.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image (JPEG, PNG, WEBP)."
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    img = _read_image(file_bytes)

    try:
        result = engine.predict_all(img=img)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-model inference failed: {str(e)}")


# Mount static assets directory
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    """Serves the unified web dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(
        content={"message": "Tomato Disease AI Dashboard API is running. Visit /docs for Swagger UI."}
    )
