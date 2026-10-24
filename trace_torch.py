import os, sys, time

# Test setting thread count and omp environment variables before torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "" # Force CPU only

print("Attempting import torch with CPU flags...", flush=True)
t0 = time.time()
try:
    import torch
    print(f"Success! Imported torch in {time.time() - t0:.2f}s, version: {torch.__version__}", flush=True)
except Exception as e:
    print(f"Failed to import torch: {e}", flush=True)
