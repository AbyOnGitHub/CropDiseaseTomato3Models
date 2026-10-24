"""
Standalone inference helper for serving/hosting.
Usage:
    from inference import predict_disease
    result = predict_disease("leaf.jpg")
"""
import json
from pathlib import Path
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
MODEL_PATH = CURRENT_DIR / "models" / "best.pt"
LABELS_PATH = CURRENT_DIR / "labels.json"

model = YOLO(str(MODEL_PATH))
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

def predict_disease(image_path: str):
    res = model(image_path, verbose=False)[0]
    top_idx = res.probs.top1
    top_conf = float(res.probs.top1conf.item())
    
    clean_label = labels[str(top_idx)]["clean_name"]
    raw_label = labels[str(top_idx)]["raw_name"]
    
    all_probs = {
        labels[str(i)]["clean_name"]: round(float(p) * 100, 2)
        for i, p in enumerate(res.probs.data.cpu().numpy())
    }
    
    return {
        "prediction": clean_label,
        "raw_label": raw_label,
        "confidence_percent": round(top_conf * 100, 2),
        "is_healthy": (clean_label.lower() == "healthy"),
        "all_probabilities": all_probs
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(json.dumps(predict_disease(sys.argv[1]), indent=4))
    else:
        print("Usage: python inference.py <image_path>")
