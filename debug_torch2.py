import os, sys, ctypes

torch_dir = r"C:\Users\abbsw\AppData\Roaming\Python\Python312\site-packages\torch"
dll_dir = os.path.join(torch_dir, "lib")
os.add_dll_directory(dll_dir)

try:
    print("Testing ctypes.CDLL(torch_python.dll)...", flush=True)
    ctypes.CDLL(os.path.join(dll_dir, "torch_python.dll"))
    print("Loaded torch_python.dll successfully!", flush=True)
except Exception as e:
    print(f"Failed loading torch_python.dll: {e}", flush=True)

try:
    print("Testing import torch._C...", flush=True)
    import torch._C
    print("torch._C imported successfully!", flush=True)
except Exception as e:
    print(f"torch._C error: {e}", flush=True)
