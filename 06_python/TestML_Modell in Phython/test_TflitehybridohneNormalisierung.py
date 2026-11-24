import os

# Muss vor dem TensorFlow-Import stehen
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


tf.get_logger().setLevel("ERROR")


# ------------------------------------------------------------
# Einstellungen
# ------------------------------------------------------------

model_path = "sign-classifier_hybrid.tflite"

image_folder = (
    r"C:\Users\nicoh\OneDrive\Desktop\Matlabcode_Teamprojekt"
    r"\09_Trainingsdaten\AusgeschnittenFürTrainingNeueVorverarbeitungMischDatenSatz"
)

supported_extensions = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp"
)


# ------------------------------------------------------------
# Dateien überprüfen
# ------------------------------------------------------------

if not os.path.isfile(model_path):
    raise FileNotFoundError(
        "Das TFLite-Modell wurde nicht gefunden:\n"
        f"{os.path.abspath(model_path)}"
    )

if not os.path.isdir(image_folder):
    raise NotADirectoryError(
        "Der Bildordner wurde nicht gefunden:\n"
        f"{image_folder}"
    )


# ------------------------------------------------------------
# TensorFlow-Lite-Modell laden
# ------------------------------------------------------------

interpreter = tf.lite.Interpreter(
    model_path=model_path
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_shape = input_details["shape"]

if len(input_shape) != 4:
    raise ValueError(
        "Es wurde ein vierdimensionaler Modelleingang "
        f"erwartet, erhalten wurde: {input_shape}"
    )

_, height, width, channels = input_shape

print("----------------------------------------")
print("TensorFlow-Lite-Modell")
print("----------------------------------------")
print("Eingabe-Shape:", input_shape)
print("Eingabe-Datentyp:", input_details["dtype"])
print("Ausgabe-Shape:", output_details["shape"])
print("Ausgabe-Datentyp:", output_details["dtype"])


# ------------------------------------------------------------
# Klassen und Bilddateien ermitteln
# ------------------------------------------------------------

class_names = sorted([
    folder_name
    for folder_name in os.listdir(image_folder)
    if os.path.isdir(
        os.path.join(image_folder, folder_name)
    )
])

if not class_names:
    raise ValueError(
        "Im Bildordner wurden keine Klassen-Unterordner gefunden."
    )

file_paths = []
true_labels = []

for class_index, class_name in enumerate(class_names):
    class_folder = os.path.join(
        image_folder,
        class_name
    )

    class_file_paths = []

    for root, _, filenames in os.walk(class_folder):
        for filename in filenames:
            if filename.lower().endswith(
                supported_extensions
            ):
                class_file_paths.append(
                    os.path.join(root, filename)
                )

    class_file_paths.sort()

    file_paths.extend(class_file_paths)
    true_labels.extend(
        [class_index] * len(class_file_paths)
    )

true_labels = np.asarray(
    true_labels,
    dtype=np.int32
)

if len(file_paths) == 0:
    raise ValueError(
        "Im Bildordner und seinen Unterordnern "
        "wurden keine Bilder gefunden."
    )

print("\nKlassenreihenfolge:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")

print(f"\nAnzahl Bilder: {len(file_paths)}")


# ------------------------------------------------------------
# Bild laden und mit Antialiasing verkleinern
# ------------------------------------------------------------

def load_and_resize_image(path):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=int(channels),
        expand_animations=False
    )

    image.set_shape([
        None,
        None,
        int(channels)
    ])

    image = tf.image.resize(
        image,
        size=(int(height), int(width)),
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=True
    )

    return image.numpy().astype(np.float32)


# ------------------------------------------------------------
# Eingabe gegebenenfalls quantisieren
# ------------------------------------------------------------

def prepare_input(image):
    input_dtype = input_details["dtype"]

    if np.issubdtype(input_dtype, np.floating):
        return np.expand_dims(
            image.astype(input_dtype),
            axis=0
        )

    scale, zero_point = input_details["quantization"]

    if scale == 0:
        raise ValueError(
            "Der Modelleingang ist ganzzahlig, besitzt aber "
            "keinen gültigen Quantisierungsfaktor."
        )

    quantized_image = np.round(
        image / scale + zero_point
    )

    limits = np.iinfo(input_dtype)

    quantized_image = np.clip(
        quantized_image,
        limits.min,
        limits.max
    ).astype(input_dtype)

    return np.expand_dims(
        quantized_image,
        axis=0
    )


# ------------------------------------------------------------
# Ausgabe gegebenenfalls dequantisieren
# ------------------------------------------------------------

def read_output():
    output = interpreter.get_tensor(
        output_details["index"]
    )

    output_dtype = output_details["dtype"]

    if np.issubdtype(output_dtype, np.integer):
        scale, zero_point = output_details["quantization"]

        if scale != 0:
            output = (
                output.astype(np.float32)
                - zero_point
            ) * scale

    return np.asarray(output).squeeze()


# ------------------------------------------------------------
# Vorhersagen für den gesamten Datensatz
# ------------------------------------------------------------

predicted_labels = []

for image_index, image_path in enumerate(
    file_paths,
    start=1
):
    image = load_and_resize_image(image_path)
    model_input = prepare_input(image)

    interpreter.set_tensor(
        input_details["index"],
        model_input
    )

    interpreter.invoke()

    scores = read_output()

    if scores.ndim != 1:
        scores = scores.reshape(-1)

    predicted_labels.append(
        int(np.argmax(scores))
    )

    if (
        image_index % 250 == 0
        or image_index == len(file_paths)
    ):
        print(
            f"Verarbeitet: {image_index}"
            f"/{len(file_paths)}"
        )

predicted_labels = np.asarray(
    predicted_labels,
    dtype=np.int32
)


# ------------------------------------------------------------
# Accuracy berechnen
# ------------------------------------------------------------

correct_predictions = (
    predicted_labels == true_labels
)

accuracy = np.mean(correct_predictions)

print(f"\nAccuracy: {accuracy * 100:.2f} %")
print(
    f"Richtig erkannt: "
    f"{np.sum(correct_predictions)} "
    f"von {len(true_labels)}"
)


# ------------------------------------------------------------
# Accuracy pro Klasse
# ------------------------------------------------------------

print("\nAccuracy je Klasse:")

for class_index, class_name in enumerate(class_names):
    class_selection = (
        true_labels == class_index
    )

    number_of_images = np.sum(
        class_selection
    )

    if number_of_images == 0:
        continue

    number_correct = np.sum(
        correct_predictions[class_selection]
    )

    class_accuracy = (
        number_correct / number_of_images
    )

    print(
        f"{class_name}: "
        f"{class_accuracy * 100:.2f} % "
        f"({number_correct}/{number_of_images})"
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

figure, axis = plt.subplots(
    figsize=(12, 10)
)

image = axis.imshow(
    confusion_matrix
)

figure.colorbar(
    image,
    ax=axis
)

axis.set(
    xticks=np.arange(number_of_classes),
    yticks=np.arange(number_of_classes),
    xticklabels=class_names,
    yticklabels=class_names,
    xlabel="Vorhergesagte Klasse",
    ylabel="Tatsächliche Klasse",
    title="Konfusionsmatrix des TensorFlow-Lite-Modells"
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