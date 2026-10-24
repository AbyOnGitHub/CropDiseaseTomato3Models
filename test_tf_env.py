import os, sys, time

torch_dir = r"C:\Users\abbsw\AppData\Roaming\Python\Python312\site-packages\torch"
dll_dir = os.path.join(torch_dir, "lib")
if os.path.exists(dll_dir):
    os.add_dll_directory(dll_dir)

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("Importing transformers...", flush=True)
t0 = time.time()
import transformers
print(f"Transformers imported in {time.time() - t0:.2f}s, version: {transformers.__version__}", flush=True)

from transformers.models.mobilevit.modeling_mobilevit import MobileViTForImageClassification
from transformers.models.mobilevit.image_processing_mobilevit import MobileViTImageProcessor
from PIL import Image
import torch

print("Loading model...", flush=True)
model = MobileViTForImageClassification.from_pretrained("MobileVIT", local_files_only=True)
processor = MobileViTImageProcessor.from_pretrained("MobileVIT", local_files_only=True)
model.eval()

img = Image.open("sample_test_leaf.jpg")
inputs = processor(images=img, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits[0], dim=0).numpy()
    pred_idx = int(probs.argmax())
    label = model.config.id2label[str(pred_idx)]
    print(f"MobileViT Predicted: {label} ({probs[pred_idx]*100:.2f}%)", flush=True)

print("MOBILEVIT WORKING 100%!", flush=True)
