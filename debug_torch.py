import os, sys

torch_dir = r"C:\Users\abbsw\AppData\Roaming\Python\Python312\site-packages\torch"
print("Torch dir exists:", os.path.exists(torch_dir), flush=True)

# check torch subdirectories and files
dll_dir = os.path.join(torch_dir, "lib")
if os.path.exists(dll_dir):
    print("Torch lib files:", [f for f in os.listdir(dll_dir) if f.endswith(".dll")][:10], flush=True)

# Check if ctypes can load dlls
import ctypes
for dll in ["c10.dll", "torch_cpu.dll"]:
    p = os.path.join(dll_dir, dll)
    if os.path.exists(p):
        try:
            print(f"Testing ctypes.CDLL({dll})...", flush=True)
            ctypes.CDLL(p)
            print(f"Loaded {dll} successfully!", flush=True)
        except Exception as e:
            print(f"Failed loading {dll}: {e}", flush=True)
