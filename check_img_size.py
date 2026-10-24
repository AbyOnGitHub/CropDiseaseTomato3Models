from PIL import Image
im = Image.open("YOLO/evaluation_charts/eda_samples.png")
print("eda_samples size:", im.size)
im2 = Image.open("YOLO/evaluation_charts/corrected_sample_predictions.png")
print("corrected_sample_predictions size:", im2.size)
