import sys, os

print("Python executable:", sys.executable)
print("Python version:", sys.version)

import site
site_packages = site.getsitepackages()
print("Site packages:", site_packages)

for sp in site_packages:
    if os.path.exists(sp):
        files = os.listdir(sp)
        torch_files = [f for f in files if 'torch' in f.lower()]
        print(f"In {sp}, torch matches: {torch_files}")
