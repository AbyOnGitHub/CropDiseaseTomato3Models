import os, sys, time

torch_dir = r"C:\Users\abbsw\AppData\Roaming\Python\Python312\site-packages\torch"
dll_dir = os.path.join(torch_dir, "lib")
if os.path.exists(dll_dir):
    os.add_dll_directory(dll_dir)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("Importing torch...", flush=True)
import torch
print("Importing transformers...", flush=True)
t0 = time.time()
from transformers import MobileViTForImageClassification, MobileViTImageProcessor
print(f"Transformers imported in {time.time() - t0:.2f}s", flush=True)

print("Loading local MobileViT...", flush=True)
t1 = time.time()
model = MobileViTForImageClassification.from_pretrained("MobileVIT", local_files_only=True)
processor = MobileViTImageProcessor.from_pretrained("MobileVIT", local_files_only=True)
model.eval()
print(f"Loaded model in {time.time() - t1:.2f}s", flush=True)

from PIL import Image
img = Image.open("sample_test_leaf.jpg")
inputs = processor(images=img, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits[0], dim=0).numpy()
    pred_idx = int(probs.argmax())
    label = model.config.id2label[str(pred_idx)]
    print(f"Prediction: {label} ({probs[pred_idx]*100:.2f}%)", flush=True)

print("SUCCESS: MobileViT works flawlessly!", flush=True)
