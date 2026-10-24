"""
Unified Model Inference Engine
Loads and runs inference on:
1. YOLO11n-cls (ONNX Runtime)
2. Novel Plant Disease CNN with Inception blocks (ONNX Runtime)
3. MobileViT Vision Transformer (PyTorch CPU with native safetensors)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image

# Ensure PyTorch on Windows can locate dependent DLLs
torch_dir = r"C:\Users\abbsw\AppData\Roaming\Python\Python312\site-packages\torch"
dll_dir = os.path.join(torch_dir, "lib")
if os.path.exists(dll_dir):
    try:
        os.add_dll_directory(dll_dir)
    except Exception:
        pass

import onnxruntime as ort
import torch
from transformers import MobileViTConfig, MobileViTForImageClassification
from safetensors.torch import load_file

from app.disease_knowledge import TOMATO_DISEASES, get_disease_info

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_ONNX_PATH = BASE_DIR / "YOLO" / "models" / "best.onnx"
NOVEL_ONNX_PATH = BASE_DIR / "Novel CNN Model" / "novel_cnn_tomato_6classes.onnx"
MOBILEVIT_CONFIG_PATH = BASE_DIR / "MobileVIT" / "config.json"
MOBILEVIT_WEIGHTS_PATH = BASE_DIR / "MobileVIT" / "model.safetensors"

# Canonical list of 6 classes
CANONICAL_CLASSES = [
    "Tomato Early Blight",
    "Tomato Healthy",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Spot",
    "Tomato Yellow Leaf Curl Virus",
]

# Model-specific class order mappings
# 1. YOLO index mapping (from YOLO/labels.json)
YOLO_INDEX_TO_CANONICAL = {
    0: "Tomato Early Blight",
    1: "Tomato Healthy",
    2: "Tomato Late Blight",
    3: "Tomato Leaf Mold",
    4: "Tomato Septoria Spot",
    5: "Tomato Yellow Leaf Curl Virus",
}

# 2. Novel CNN index mapping (from notebook cell 11: CANONICAL_TOMATO_CLASSES)
NOVEL_CNN_INDEX_TO_CANONICAL = {
    0: "Tomato Early Blight",
    1: "Tomato Late Blight",
    2: "Tomato Septoria Spot",
    3: "Tomato Leaf Mold",
    4: "Tomato Yellow Leaf Curl Virus",
    5: "Tomato Healthy",
}

# 3. MobileViT index mapping (from MobileVIT/config.json id2label)
MOBILEVIT_INDEX_TO_CANONICAL = {
    0: "Tomato Early Blight",
    1: "Tomato Healthy",
    2: "Tomato Late Blight",
    3: "Tomato Leaf Mold",
    4: "Tomato Septoria Spot",
    5: "Tomato Yellow Leaf Curl Virus",
}


class ModelEngine:
    """Manages loading and running all three tomato disease classification models."""

    def __init__(self):
        self.yolo_session = None
        self.novel_session = None
        self.mobilevit_model = None
        self.mobilevit_config = None
        self.is_loaded = False

    def load_models(self):
        """Loads all three models into memory."""
        print("[*] Initializing Unified Model Engine...", flush=True)

        # 1. Load YOLO ONNX Session
        print(f"[*] Loading YOLO11n-cls from {YOLO_ONNX_PATH} ...", flush=True)
        t0 = time.time()
        self.yolo_session = ort.InferenceSession(
            str(YOLO_ONNX_PATH),
            providers=["CPUExecutionProvider"]
        )
        print(f"[+] YOLO11n-cls loaded in {(time.time() - t0)*1000:.1f}ms", flush=True)

        # 2. Load Novel CNN ONNX Session
        print(f"[*] Loading Novel CNN from {NOVEL_ONNX_PATH} ...", flush=True)
        t0 = time.time()
        self.novel_session = ort.InferenceSession(
            str(NOVEL_ONNX_PATH),
            providers=["CPUExecutionProvider"]
        )
        print(f"[+] Novel CNN loaded in {(time.time() - t0)*1000:.1f}ms", flush=True)

        # 3. Load MobileViT PyTorch Model
        print(f"[*] Loading MobileViT from {MOBILEVIT_WEIGHTS_PATH} ...", flush=True)
        t0 = time.time()
        self.mobilevit_config = MobileViTConfig.from_json_file(str(MOBILEVIT_CONFIG_PATH))
        self.mobilevit_model = MobileViTForImageClassification(self.mobilevit_config)
        state_dict = load_file(str(MOBILEVIT_WEIGHTS_PATH))
        self.mobilevit_model.load_state_dict(state_dict)
        self.mobilevit_model.eval()
        print(f"[+] MobileViT loaded in {(time.time() - t0)*1000:.1f}ms", flush=True)

        self.is_loaded = True
        print("[+] All 3 classification models are loaded and ready!", flush=True)

    def get_models_metadata(self) -> List[Dict[str, Any]]:
        """Returns metadata specifications for all 3 models."""
        return [
            {
                "id": "yolo",
                "name": "Ultralytics YOLO11n-cls",
                "architecture": "YOLO11 Nano Classification",
                "type": "Convolutional Anchor-Free Vision Model",
                "framework": "Ultralytics YOLO / ONNX Runtime",
                "parameters": "3,197,890 (3.2M)",
                "input_resolution": "256 × 256 × 3",
                "test_accuracy": "99.8%",
                "badge": "Fastest & Top Accuracy",
                "speed_rating": "Ultra-Fast (~15ms)",
                "recommended_for": "Real-time edge & field deployment with highest overall accuracy."
            },
            {
                "id": "novel_cnn",
                "name": "Novel Plant Disease CNN",
                "architecture": "Inception-A/B + Depthwise Separable CNN",
                "type": "Custom Multi-Scale Residual CNN",
                "framework": "PyTorch / ONNX Runtime",
                "parameters": "597,510 (~597K)",
                "input_resolution": "256 × 256 × 3",
                "test_accuracy": "98.2%",
                "badge": "Ultra-Lightweight Inception",
                "speed_rating": "Fast (~20ms)",
                "recommended_for": "Embedded microcontrollers and low-power IoT farm monitors."
            },
            {
                "id": "mobilevit",
                "name": "MobileViT Vision Transformer",
                "architecture": "apple/mobilevit-xx-small (Transformer + CNN)",
                "type": "Hybrid Vision Transformer",
                "framework": "Hugging Face / PyTorch",
                "parameters": "1,300,000 (1.3M)",
                "input_resolution": "256 × 256 × 3",
                "test_accuracy": "98.6%",
                "badge": "Global Attention Transformer",
                "speed_rating": "Fast (~90ms)",
                "recommended_for": "Fine-grained visual texture analysis and global context modeling."
            }
        ]

    # --- Preprocessing Routines ---

    @staticmethod
    def _preprocess_yolo(img: Image.Image) -> np.ndarray:
        """Preprocesses image for YOLO: 256x256, scaled [0, 1], shape (1, 3, 256, 256)."""
        resized = img.resize((256, 256), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1)) # HWC -> CHW
        return np.expand_dims(arr, axis=0)

    @staticmethod
    def _preprocess_novel_cnn(img: Image.Image) -> np.ndarray:
        """Preprocesses image for Novel CNN: 256x256, ImageNet normalization, shape (1, 3, 256, 256)."""
        resized = img.resize((256, 256), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0).astype(np.float32)

    @staticmethod
    def _preprocess_mobilevit(img: Image.Image) -> torch.Tensor:
        """Preprocesses image for MobileViT: resize shortest edge to 288, center crop 256x256, normalize (x-0.5)/0.5."""
        w, h = img.size
        if w < h:
            new_w, new_h = 288, int(h * 288 / w)
        else:
            new_w, new_h = int(w * 288 / h), 288
        img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        left = (new_w - 256) // 2
        top = (new_h - 256) // 2
        img_cropped = img_resized.crop((left, top, left + 256, top + 256))

        arr = np.array(img_cropped, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)
        return torch.from_numpy(arr).float()

    # --- Inference Methods ---

    def predict_yolo(self, img: Image.Image) -> Dict[str, Any]:
        """Runs inference via YOLO11n-cls ONNX model."""
        t_start = time.time()
        input_tensor = self._preprocess_yolo(img)
        input_name = self.yolo_session.get_inputs()[0].name
        raw_outputs = self.yolo_session.run(None, {input_name: input_tensor})
        logits = raw_outputs[0][0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()

        pred_idx = int(np.argmax(probabilities))
        pred_canonical = YOLO_INDEX_TO_CANONICAL[pred_idx]
        confidence = float(probabilities[pred_idx])
        latency_ms = (time.time() - t_start) * 1000

        # Build full class distribution in canonical format
        class_probs = []
        for idx in range(len(probabilities)):
            canonical_name = YOLO_INDEX_TO_CANONICAL[idx]
            prob = float(probabilities[idx])
            class_probs.append({
                "class_name": canonical_name,
                "probability": prob,
                "percentage": round(prob * 100, 2)
            })

        class_probs.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "model_id": "yolo",
            "model_name": "Ultralytics YOLO11n-cls",
            "predicted_class": pred_canonical,
            "confidence_score": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "is_healthy": (pred_canonical == "Tomato Healthy"),
            "latency_ms": round(latency_ms, 1),
            "probabilities": class_probs,
            "disease_info": get_disease_info(pred_canonical)
        }

    def predict_novel_cnn(self, img: Image.Image) -> Dict[str, Any]:
        """Runs inference via Novel Plant Disease CNN ONNX model."""
        t_start = time.time()
        input_tensor = self._preprocess_novel_cnn(img)
        input_name = self.novel_session.get_inputs()[0].name
        raw_outputs = self.novel_session.run(None, {input_name: input_tensor})
        logits = raw_outputs[0][0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()

        pred_idx = int(np.argmax(probabilities))
        pred_canonical = NOVEL_CNN_INDEX_TO_CANONICAL[pred_idx]
        confidence = float(probabilities[pred_idx])
        latency_ms = (time.time() - t_start) * 1000

        class_probs = []
        for idx in range(len(probabilities)):
            canonical_name = NOVEL_CNN_INDEX_TO_CANONICAL[idx]
            prob = float(probabilities[idx])
            class_probs.append({
                "class_name": canonical_name,
                "probability": prob,
                "percentage": round(prob * 100, 2)
            })

        class_probs.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "model_id": "novel_cnn",
            "model_name": "Novel Plant Disease CNN",
            "predicted_class": pred_canonical,
            "confidence_score": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "is_healthy": (pred_canonical == "Tomato Healthy"),
            "latency_ms": round(latency_ms, 1),
            "probabilities": class_probs,
            "disease_info": get_disease_info(pred_canonical)
        }

    def predict_mobilevit(self, img: Image.Image) -> Dict[str, Any]:
        """Runs inference via fine-tuned MobileViT PyTorch model."""
        t_start = time.time()
        tensor_input = self._preprocess_mobilevit(img)

        with torch.no_grad():
            outputs = self.mobilevit_model(pixel_values=tensor_input)
            probs = torch.softmax(outputs.logits[0], dim=0).numpy()

        pred_idx = int(np.argmax(probs))
        pred_canonical = MOBILEVIT_INDEX_TO_CANONICAL[pred_idx]
        confidence = float(probs[pred_idx])
        latency_ms = (time.time() - t_start) * 1000

        class_probs = []
        for idx in range(len(probs)):
            canonical_name = MOBILEVIT_INDEX_TO_CANONICAL[idx]
            prob = float(probs[idx])
            class_probs.append({
                "class_name": canonical_name,
                "probability": prob,
                "percentage": round(prob * 100, 2)
            })

        class_probs.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "model_id": "mobilevit",
            "model_name": "MobileViT Vision Transformer",
            "predicted_class": pred_canonical,
            "confidence_score": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "is_healthy": (pred_canonical == "Tomato Healthy"),
            "latency_ms": round(latency_ms, 1),
            "probabilities": class_probs,
            "disease_info": get_disease_info(pred_canonical)
        }

    def predict(self, model_id: str, img: Image.Image) -> Dict[str, Any]:
        """Dispatch prediction request to chosen model."""
        if not self.is_loaded:
            self.load_models()

        clean_id = model_id.lower().strip()
        if clean_id in ["yolo", "yolo11", "yolo11n"]:
            return self.predict_yolo(img)
        elif clean_id in ["novel_cnn", "novel", "inception"]:
            return self.predict_novel_cnn(img)
        elif clean_id in ["mobilevit", "vit", "transformer"]:
            return self.predict_mobilevit(img)
        else:
            raise ValueError(f"Unknown model_id: '{model_id}'. Choose from: 'yolo', 'novel_cnn', 'mobilevit'.")

    def predict_all(self, img: Image.Image) -> Dict[str, Any]:
        """Runs all 3 models simultaneously on the same image and computes consensus."""
        if not self.is_loaded:
            self.load_models()

        res_yolo = self.predict_yolo(img)
        res_novel = self.predict_novel_cnn(img)
        res_mobilevit = self.predict_mobilevit(img)

        results = {
            "yolo": res_yolo,
            "novel_cnn": res_novel,
            "mobilevit": res_mobilevit,
        }

        # Calculate vote consensus
        votes: Dict[str, int] = {}
        weighted_conf: Dict[str, float] = {}

        for r in [res_yolo, res_novel, res_mobilevit]:
            c = r["predicted_class"]
            votes[c] = votes.get(c, 0) + 1
            weighted_conf[c] = weighted_conf.get(c, 0.0) + r["confidence_score"]

        # Majority or highest combined confidence
        sorted_candidates = sorted(
            votes.keys(),
            key=lambda k: (votes[k], weighted_conf[k]),
            reverse=True
        )
        consensus_class = sorted_candidates[0]
        consensus_agreement = f"{votes[consensus_class]}/3 Models Agree"

        return {
            "consensus_class": consensus_class,
            "consensus_agreement": consensus_agreement,
            "consensus_disease_info": get_disease_info(consensus_class),
            "is_healthy": (consensus_class == "Tomato Healthy"),
            "models_comparison": results
        }


# Singleton engine instance
engine = ModelEngine()
