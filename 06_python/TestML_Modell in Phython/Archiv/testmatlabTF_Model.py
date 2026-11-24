import os

# Muss vor dem TensorFlow-Import stehen
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# MATLAB-Converter-Package
import tf_model


tf.get_logger().setLevel("ERROR")


# ------------------------------------------------------------
# TensorFlow-Modell laden
# ------------------------------------------------------------

model = tf_model.load_model(
    load_weights=True,
    debug=False
)

print("Eingabe-Shape:", model.input_shape)
print("Ausgabe-Shape:", model.output_shape)


# ------------------------------------------------------------
# Ordner mit Bildern
# ------------------------------------------------------------

image_folder = (
    r"C:\Users\nicoh\OneDrive\Desktop\Matlabcode_Teamprojekt"
    r"\09_Trainingsdaten\AusgeschnittenFuerTraining"
    r"\99_Trainingsdatensaetze"
    r"\03_13Kat_Alle Schilder"
)


# ------------------------------------------------------------
# Bilder aus allen Klassen-Unterordnern laden
# ------------------------------------------------------------

_, height, width, channels = model.input_shape

dataset = tf.keras.utils.image_dataset_from_directory(
    image_folder,
    labels="inferred",
    label_mode="int",
    image_size=(height, width),
    color_mode="grayscale" if channels == 1 else "rgb",
    batch_size=32,
    shuffle=False
)

class_names = dataset.class_names
file_paths = dataset.file_paths

print("\nKlassenreihenfolge:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


# ------------------------------------------------------------
# Tatsächliche Klassen aus dem Dataset auslesen
# ------------------------------------------------------------

true_labels = np.concatenate([
    labels.numpy()
    for images, labels in dataset
])

print(f"\nAnzahl Bilder: {len(true_labels)}")


# ------------------------------------------------------------
# Nur die Bilder an das Modell übergeben
# ------------------------------------------------------------

image_dataset = dataset.map(
    lambda images, labels: images
)


# ------------------------------------------------------------
# Vorhersagen für den gesamten Datensatz
# ------------------------------------------------------------

predictions = model.predict(
    image_dataset,
    verbose=1
)

# Falls das Modell mehrere Ausgaben besitzt
if isinstance(predictions, (list, tuple)):
    predictions = predictions[0]

predictions = np.asarray(predictions)
predictions = np.squeeze(predictions)


# Bei nur einem Bild die Batch-Dimension wiederherstellen
if predictions.ndim == 1:
    predictions = np.expand_dims(predictions, axis=0)


# Falls MATLAB/TensorFlow die Dimensionen vertauscht ausgegeben hat
if (
    predictions.shape[0] != len(true_labels)
    and predictions.shape[1] == len(true_labels)
):
    predictions = predictions.T


print("\nForm der Modellausgabe:", predictions.shape)


# ------------------------------------------------------------
# Vorhergesagte Klassen bestimmen
# ------------------------------------------------------------

predicted_labels = np.argmax(
    predictions,
    axis=1
)


# ------------------------------------------------------------
# Accuracy berechnen
# ------------------------------------------------------------

correct_predictions = predicted_labels == true_labels

accuracy = np.mean(correct_predictions)

print(f"\nAccuracy: {accuracy * 100:.2f} %")
print(
    f"Richtig erkannt: {np.sum(correct_predictions)} "
    f"von {len(true_labels)}"
)


# ------------------------------------------------------------
# Accuracy pro Klasse
# ------------------------------------------------------------

print("\nAccuracy je Klasse:")

for class_index, class_name in enumerate(class_names):

    class_selection = true_labels == class_index
    number_of_images = np.sum(class_selection)

    if number_of_images == 0:
        continue

    class_accuracy = np.mean(
        predicted_labels[class_selection]
        == true_labels[class_selection]
    )

    print(
        f"{class_name}: "
        f"{class_accuracy * 100:.2f} % "
        f"({np.sum(correct_predictions[class_selection])}"
        f"/{number_of_images})"
    )


# ------------------------------------------------------------
# Konfusionsmatrix berechnen und anzeigen
# ------------------------------------------------------------

number_of_classes = len(class_names)

confusion_matrix = tf.math.confusion_matrix(
    true_labels,
    predicted_labels,
    num_classes=number_of_classes
).numpy()

figure, axis = plt.subplots(figsize=(12, 10))

image = axis.imshow(confusion_matrix)
figure.colorbar(image, ax=axis)

axis.set(
    xticks=np.arange(number_of_classes),
    yticks=np.arange(number_of_classes),
    xticklabels=class_names,
    yticklabels=class_names,
    xlabel="Vorhergesagte Klasse",
    ylabel="Tatsächliche Klasse",
    title="Konfusionsmatrix des TensorFlow-Modells"
)

plt.setp(
    axis.get_xticklabels(),
    rotation=90,
    ha="center"
)

for row in range(number_of_classes):
    for column in range(number_of_classes):
        axis.text(
            column,
            row,
            str(confusion_matrix[row, column]),
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()
