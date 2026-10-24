import os
from PIL import Image

os.makedirs("static/samples", exist_ok=True)

# Let's inspect corrected_sample_predictions.png
img = Image.open("YOLO/evaluation_charts/corrected_sample_predictions.png")
w, h = img.size
print(f"Sample predictions chart size: {w}x{h}")

# In a typical matplotlib grid of predictions, we have subplots.
# Let's crop 6 distinct sub-regions or use sample patches:
# Let's crop 6 sample patches of size 350x350 from diverse locations in the chart
crops = [
    ("sample_1_early_blight.jpg", (150, 150, 500, 500)),
    ("sample_2_healthy.jpg", (800, 150, 1150, 500)),
    ("sample_3_late_blight.jpg", (1500, 150, 1850, 500)),
    ("sample_4_leaf_mold.jpg", (2200, 150, 2550, 500)),
    ("sample_5_septoria_spot.jpg", (2900, 150, 3250, 500)),
    ("sample_6_yellow_curl.jpg", (3600, 150, 3950, 500)),
]

for filename, box in crops:
    patch = img.crop(box).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    target_path = os.path.join("static", "samples", filename)
    patch.save(target_path, quality=95)
    print(f"[+] Saved sample: {target_path}")

print("[+] All sample images created successfully!")
