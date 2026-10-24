from safetensors import safe_open

with safe_open("MobileVIT/model.safetensors", framework="numpy") as f:
    keys = list(f.keys())
    print(f"Total keys in safetensors: {len(keys)}")
    for k in keys[:10]:
        t = f.get_tensor(k)
        print(f"  {k}: shape {t.shape}, dtype {t.dtype}")
