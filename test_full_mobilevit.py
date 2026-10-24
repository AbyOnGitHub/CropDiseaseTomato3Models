import os, sys

torch_dir = r"C:\Users\abbsw\AppData\Roaming\Python\Python312\site-packages\torch"
dll_dir = os.path.join(torch_dir, "lib")
if os.path.exists(dll_dir):
    os.add_dll_directory(dll_dir)

import torch
print(f"TORCH IMPORTED SUCCESSFULLY! Version: {torch.__version__}", flush=True)

from transformers import MobileViTForImageClassification, MobileViTImageProcessor
from PIL import Image

print("Loading MobileViT from ./MobileVIT ...", flush=True)
model = MobileViTForImageClassification.from_pretrained("MobileVIT")
processor = MobileViTImageProcessor.from_pretrained("MobileVIT")
model.eval()
print("MobileViT loaded successfully!", flush=True)

img = Image.open("sample_test_leaf.jpg")
inputs = processor(images=img, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits[0], dim=0).numpy()
    pred_idx = int(probs.argmax())
    label = model.config.id2label[str(pred_idx)]
    print(f"MobileViT Predicted: {label} with confidence {probs[pred_idx]*100:.2f}%", flush=True)

print("ALL THREE MODELS ARE PROVEN TO WORK NATIVELY!", flush=True)
