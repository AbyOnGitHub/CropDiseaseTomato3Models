
import os
import sys
import numpy as np
import tensorflow as tf
import onnx2tf

crop_name = 'tomato_6classes'
onnx_path = 'novel_cnn_tomato_6classes.onnx'
tf_saved_model_dir = 'saved_model_tomato_6classes'
tflite_fp32_path = 'novel_cnn_tomato_6classes_fp32.tflite'
tflite_fp16_path = 'novel_cnn_tomato_6classes_fp16.tflite'

# 1. Convert ONNX -> SavedModel
onnx2tf.convert(
    input_onnx_file_path=onnx_path,
    output_folder_path=tf_saved_model_dir,
    copy_onnx_input_output_names_to_tflite=True,
    non_verbose=True
)

# 2. Convert to FP32 TFLite
converter_fp32 = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)
tflite_fp32_bytes = converter_fp32.convert()
with open(tflite_fp32_path, 'wb') as f:
    f.write(tflite_fp32_bytes)

# 3. Convert to Float16 Quantized TFLite (Half size, optimal for mobile/edge)
converter_fp16 = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)
converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_fp16.target_spec.supported_types = [tf.float16]
tflite_fp16_bytes = converter_fp16.convert()
with open(tflite_fp16_path, 'wb') as f:
    f.write(tflite_fp16_bytes)

print(f'   --> Created FP32 TFLite: {tflite_fp32_path} ({len(tflite_fp32_bytes)/(1024*1024):.2f} MB)', flush=True)
print(f'   --> Created FP16 TFLite: {tflite_fp16_path} ({len(tflite_fp16_bytes)/(1024*1024):.2f} MB)', flush=True)

# 4. Verify TFLite Interpreter
interpreter = tf.lite.Interpreter(model_path=tflite_fp32_path)
interpreter.allocate_tensors()
in_details = interpreter.get_input_details()
out_details = interpreter.get_output_details()

dummy_tflite_in = np.random.randn(*in_details[0]['shape']).astype(np.float32)
interpreter.set_tensor(in_details[0]['index'], dummy_tflite_in)
interpreter.invoke()
out_logits = interpreter.get_tensor(out_details[0]['index'])
print(f'   --> TFLite Inference Test Passed! Output logits shape: {out_logits.shape}', flush=True)
