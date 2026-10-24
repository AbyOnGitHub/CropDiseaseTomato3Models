import os
import json
import numpy as np
from PIL import Image

# 1. Create a sample test image
img = Image.new("RGB", (256, 256), color=(34, 139, 34)) # Forest green
sample_img_path = "sample_test_leaf.jpg"
img.save(sample_img_path)
print(f"[+] Created test image: {sample_img_path}", flush=True)

# 2. Test YOLO (Ultralytics & ONNX)
print("\n--- Testing Model 1: YOLO ---", flush=True)
try:
    import onnxruntime as ort
    yolo_onnx_path = os.path.join("YOLO", "models", "best.onnx")
    session = ort.InferenceSession(yolo_onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"YOLO ONNX input: {input_name}, shape: {input_shape}", flush=True)
    
    # Preprocess image for YOLO (usually 1, 3, 256, 256 normalized 0-1)
    im_resized = img.resize((256, 256))
    im_arr = np.array(im_resized, dtype=np.float32) / 255.0
    im_arr = np.transpose(im_arr, (2, 0, 1)) # HWC to CHW
    im_arr = np.expand_dims(im_arr, axis=0) # Add batch dim (1, 3, 256, 256)
    
    outputs = session.run(None, {input_name: im_arr})
    logits = outputs[0][0]
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    pred_idx = int(np.argmax(probs))
    print(f"YOLO ONNX Pred Index: {pred_idx}, confidence: {probs[pred_idx]:.4f}", flush=True)
    print("YOLO ONNX inference succeeded!", flush=True)
except Exception as e:
    print(f"YOLO ONNX Error: {e}", flush=True)

# 3. Test Novel CNN Model (ONNX)
print("\n--- Testing Model 2: Novel CNN ---", flush=True)
try:
    novel_onnx_path = os.path.join("Novel CNN Model", "novel_cnn_tomato_6classes.onnx")
    session_novel = ort.InferenceSession(novel_onnx_path, providers=['CPUExecutionProvider'])
    in_name = session_novel.get_inputs()[0].name
    in_shape = session_novel.get_inputs()[0].shape
    print(f"Novel CNN ONNX input: {in_name}, shape: {in_shape}", flush=True)
    
    # Preprocessing based on notebook:
    # Resize 256x256, ToTensor (0-1), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    im_arr = np.array(img.resize((256, 256)), dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    im_arr = (im_arr - mean) / std
    im_arr = np.transpose(im_arr, (2, 0, 1))
    im_arr = np.expand_dims(im_arr, axis=0).astype(np.float32)
    
    outputs = session_novel.run(None, {in_name: im_arr})
    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    pred_idx = int(np.argmax(probs))
    print(f"Novel CNN Pred Index: {pred_idx}, confidence: {probs[pred_idx]:.4f}", flush=True)
    print("Novel CNN ONNX inference succeeded!", flush=True)
except Exception as e:
    print(f"Novel CNN Error: {e}", flush=True)

# 4. Test MobileViT
print("\n--- Testing Model 3: MobileViT ---", flush=True)
try:
    import torch
    from transformers import MobileViTForImageClassification, MobileViTImageProcessor
    mobilevit_dir = "MobileVIT"
    model = MobileViTForImageClassification.from_pretrained(mobilevit_dir)
    processor = MobileViTImageProcessor.from_pretrained(mobilevit_dir)
    model.eval()
    
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=0).numpy()
        pred_idx = int(np.argmax(probs))
        label = model.config.id2label[str(pred_idx)]
    print(f"MobileViT Pred: {label} (Idx {pred_idx}), confidence: {probs[pred_idx]:.4f}", flush=True)
    print("MobileViT inference succeeded!", flush=True)
except Exception as e:
    print(f"MobileViT Error: {e}", flush=True)
