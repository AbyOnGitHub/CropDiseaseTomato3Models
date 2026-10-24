import time, importlib.metadata
t0 = time.time()
print("Starting packages_distributions()...", flush=True)
res = importlib.metadata.packages_distributions()
print(f"packages_distributions took: {time.time() - t0:.2f}s, found {len(res)} mappings", flush=True)
