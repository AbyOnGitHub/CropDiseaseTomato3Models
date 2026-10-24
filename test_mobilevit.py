import time, os
print("Starting MobileViT test...", flush=True)
t0 = time.time()

import torch
print(f"torch imported in {time.time() - t0:.2f}s, torch version: {torch.__version__}", flush=True)

from transformers import MobileViTForImageClassification, MobileViTImageProcessor
from PIL import Image

mobilevit_dir = "MobileVIT"
print("Loading model and processor...", flush=True)
t1 = time.time()
model = MobileViTForImageClassification.from_pretrained(mobilevit_dir)
processor = MobileViTImageProcessor.from_pretrained(mobilevit_dir)
model.eval()
print(f"Model loaded in {time.time() - t1:.2f}s", flush=True)

img = Image.open("sample_test_leaf.jpg")
inputs = processor(images=img, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=0).numpy()
    pred_idx = int(probs.argmax())
    label = model.config.id2label[str(pred_idx)]
    print(f"Prediction: {label} (confidence: {probs[pred_idx]*100:.2f}%)", flush=True)
    for idx, p in enumerate(probs):
        print(f"  {model.config.id2label[str(idx)]}: {p*100:.2f}%", flush=True)
print("MobileViT TEST PASSED SUCCESSFULLY!", flush=True)
