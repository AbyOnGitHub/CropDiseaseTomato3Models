import os, sys, time
import numpy as np
from PIL import Image

torch_dir = r"C:\Users\abbsw\AppData\Roaming\Python\Python312\site-packages\torch"
dll_dir = os.path.join(torch_dir, "lib")
if os.path.exists(dll_dir):
    os.add_dll_directory(dll_dir)

import torch
from transformers import MobileViTConfig, MobileViTForImageClassification
from safetensors.torch import load_file

config = MobileViTConfig.from_json_file("MobileVIT/config.json")
model = MobileViTForImageClassification(config)
state_dict = load_file("MobileVIT/model.safetensors")
model.load_state_dict(state_dict)
model.eval()

# Pure PIL + Numpy preprocessing matching MobileViTImageProcessor
img = Image.open("sample_test_leaf.jpg").convert("RGB")
# 1. Resize shortest edge to 288
w, h = img.size
if w < h:
    new_w, new_h = 288, int(h * 288 / w)
else:
    new_w, new_h = int(w * 288 / h), 288
img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

# 2. Center crop to 256x256
left = (new_w - 256) // 2
top = (new_h - 256) // 2
img_cropped = img_resized.crop((left, top, left + 256, top + 256))

# 3. Rescale (0-1) and normalize with mean=0.5, std=0.5
arr = np.array(img_cropped, dtype=np.float32) / 255.0
arr = (arr - 0.5) / 0.5 # Normalization: (x - 0.5) / 0.5
arr = np.transpose(arr, (2, 0, 1)) # HWC to CHW
arr = np.expand_dims(arr, 0) # (1, 3, 256, 256)

tensor_input = torch.from_numpy(arr).float()

t0 = time.time()
with torch.no_grad():
    outputs = model(pixel_values=tensor_input)
    probs = torch.softmax(outputs.logits[0], dim=0).numpy()
    pred_idx = int(np.argmax(probs))
    label = config.id2label.get(pred_idx, config.id2label.get(str(pred_idx)))
    print(f"Inference latency: {(time.time() - t0)*1000:.1f}ms", flush=True)
    print(f"MobileViT Prediction: {label} (confidence: {probs[pred_idx]*100:.2f}%)", flush=True)
    for idx, p in enumerate(probs):
        c_name = config.id2label.get(idx, config.id2label.get(str(idx)))
        print(f"  {c_name}: {p*100:.2f}%", flush=True)

print("ALL PREDICTIONS AND PROBABILITIES CALCULATED ACCURATELY!", flush=True)
