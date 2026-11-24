import os
import tensorflow as tf

model_path = "sign-classifier_hybrid.tflite"

print("Datei vorhanden:", os.path.isfile(model_path))
print("Dateigröße:", os.path.getsize(model_path), "Bytes")

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

print("TFLite-Modell erfolgreich geladen")
print("Eingabe:", interpreter.get_input_details())
print("Ausgabe:", interpreter.get_output_details())