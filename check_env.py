import sys

modules = ['torch', 'torchvision', 'ultralytics', 'transformers', 'onnxruntime', 'fastapi', 'uvicorn', 'PIL', 'safetensors', 'numpy']

for m in modules:
    try:
        mod = __import__(m)
        print(f"{m}: OK (version: {getattr(mod, '__version__', 'unknown')})", flush=True)
    except Exception as e:
        print(f"{m}: NOT FOUND / ERROR: {e}", flush=True)
