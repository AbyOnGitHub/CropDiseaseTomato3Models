import sys
from PIL import Image
from app.model_engine import engine

print("[*] Testing ModelEngine load...", flush=True)
engine.load_models()

img = Image.open("sample_test_leaf.jpg")

print("\n--- 1. Testing YOLO prediction ---", flush=True)
y_res = engine.predict("yolo", img)
print(f"YOLO -> {y_res['predicted_class']} ({y_res['confidence_percent']}%, {y_res['latency_ms']}ms)", flush=True)

print("\n--- 2. Testing Novel CNN prediction ---", flush=True)
n_res = engine.predict("novel_cnn", img)
print(f"Novel CNN -> {n_res['predicted_class']} ({n_res['confidence_percent']}%, {n_res['latency_ms']}ms)", flush=True)

print("\n--- 3. Testing MobileViT prediction ---", flush=True)
m_res = engine.predict("mobilevit", img)
print(f"MobileViT -> {m_res['predicted_class']} ({m_res['confidence_percent']}%, {m_res['latency_ms']}ms)", flush=True)

print("\n--- 4. Testing Predict All (Consensus) ---", flush=True)
all_res = engine.predict_all(img)
print(f"Consensus: {all_res['consensus_class']} ({all_res['consensus_agreement']})", flush=True)

print("\n[+] ALL 3 MODELS TESTED AND VERIFIED WORKING NATIVELY!", flush=True)
