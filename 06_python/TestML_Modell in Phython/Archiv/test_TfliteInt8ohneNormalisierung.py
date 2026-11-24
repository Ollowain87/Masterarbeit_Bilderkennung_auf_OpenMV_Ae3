import numpy as np
import tensorflow as tf
from PIL import Image
import os

interpreter = tf.lite.Interpreter(model_path="sign-classifier_alt.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

_, height, width, channels = input_details[0]["shape"]

print("Input:", input_details) 
details = interpreter.get_tensor_details()
for d in details: print(d["name"], d["dtype"], d["quantization"])
print("Output:", output_details) 
print("Quantization:", input_details[0]["quantization"])

image_folder = "Quantisierungsbilder"

for fname in os.listdir(image_folder):
    if not fname.lower().endswith((".jpg", ".png")):
        continue

    img = Image.open(os.path.join(image_folder, fname))

    if channels == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    img = img.resize((width, height))

    # FLOAT32 direkt verwenden
    arr = np.array(img, dtype=np.float32)

    if channels == 1:
        arr = np.expand_dims(arr, axis=-1)

    # Quantisierungsparameter holen 
    scale, zero_point = input_details[0]["quantization"]

    # In INT8 quantisieren 
    arr_q = arr / scale + zero_point
    arr_q = np.clip(arr_q, -128, 127).astype(np.int8)

    batch = np.expand_dims(arr_q, axis=0)

    interpreter.set_tensor(input_details[0]["index"], batch)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])
    class_idx = np.argmax(preds, axis=-1)[0][0]

    print(f"{fname} → Output={preds[0]}, Klasse={class_idx}")

