import os, sys, time

torch_dir = r"C:\Users\abbsw\AppData\Roaming\Python\Python312\site-packages\torch"
dll_dir = os.path.join(torch_dir, "lib")
if os.path.exists(dll_dir):
    os.add_dll_directory(dll_dir)

import torch
print("Torch imported successfully!", flush=True)

# Test importing only mobilevit
t0 = time.time()
print("Importing MobileViT model definition...", flush=True)
from transformers import MobileViTConfig, MobileViTForImageClassification
print(f"Imported in {time.time() - t0:.2f}s", flush=True)

config = MobileViTConfig.from_json_file("MobileVIT/config.json")
print("Instantiating model from config...", flush=True)
model = MobileViTForImageClassification(config)

print("Loading weights from safetensors...", flush=True)
from safetensors.torch import load_file
state_dict = load_file("MobileVIT/model.safetensors")
model.load_state_dict(state_dict)
model.eval()
print("MobileViT instantiated & loaded directly without full transformers overhead!", flush=True)

# Test dummy forward pass
dummy = torch.randn(1, 3, 256, 256)
out = model(dummy)
logits = out.logits
print("Logits shape:", logits.shape, flush=True)

# Now export to ONNX so inference is lightning fast!
onnx_path = "MobileVIT/mobilevit.onnx"
print(f"Exporting to {onnx_path}...", flush=True)
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["input"],
    output_names=["logits"],
    opset_version=14,
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
)
print("ONNX export completed!", flush=True)
