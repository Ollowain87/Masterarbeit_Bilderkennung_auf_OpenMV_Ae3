import os

# Muss vor dem TensorFlow-Import stehen
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from scipy.io import loadmat

# MATLAB-Converter-Package
import tf_model


tf.get_logger().setLevel("ERROR")


# ------------------------------------------------------------
# Einstellungen
# ------------------------------------------------------------

# Einzelnes Bild, das mit dem MATLAB-Ergebnis verglichen wird
image_path = (
    r"C:\Users\nicoh\OneDrive\Desktop\Matlabcode_Teamprojekt"
    r"\09_Trainingsdaten\AusgeschnittenFuerTraining"
    r"\99_Trainingsdatensaetze"
    r"\03_13Kat_Alle Schilder"
    r"\Geschwindigkeitsbegrenzung_100"
    r"\Geschwindigkeitsbegrenzung_100_136.jpg"
)

# Oberster Ordner mit allen Klassen-Unterordnern
image_folder = (
    r"C:\Users\nicoh\OneDrive\Desktop\Matlabcode_Teamprojekt"
    r"\09_Trainingsdaten\AusgeschnittenFuerTraining"
    r"\99_Trainingsdatensaetze"
    r"\03_13Kat_Alle Schilder"
)

# Von MATLAB erzeugte Vergleichsdatei
comparison_file = (
    r"C:\Users\nicoh\OneDrive\Desktop\Matlabcode_Teamprojekt"
    r"\06_python\TestML_Modell in Phython"
    r"\unabhaengiger_bildvergleich.mat"
)


# ------------------------------------------------------------
# Dateien überprüfen
# ------------------------------------------------------------

if not os.path.isfile(image_path):
    raise FileNotFoundError(
        "Das Vergleichsbild wurde nicht gefunden:\n"
        f"{image_path}"
    )

if not os.path.isdir(image_folder):
    raise NotADirectoryError(
        "Der Bildordner wurde nicht gefunden:\n"
        f"{image_folder}"
    )

if not os.path.isfile(comparison_file):
    raise FileNotFoundError(
        "Die MATLAB-Vergleichsdatei wurde nicht gefunden:\n"
        f"{comparison_file}"
    )


# ------------------------------------------------------------
# TensorFlow-Modell laden
# ------------------------------------------------------------

model = tf_model.load_model(
    load_weights=True,
    debug=False
)

print("----------------------------------------")
print("TensorFlow-Modell")
print("----------------------------------------")
print("Eingabe-Shape:", model.input_shape)
print("Ausgabe-Shape:", model.output_shape)


# ------------------------------------------------------------
# Klassenreihenfolge und Dateipfade wie im ursprünglichen Testskript ermitteln
# ------------------------------------------------------------

_, height, width, channels = model.input_shape

# Explizite Einstellungen für die Größenanpassung
USE_ANTIALIAS = True

# Zunächst False verwenden, um nur den Einfluss des Antialiasings
# zu untersuchen. Danach optional auf True setzen, um zusätzlich
# die ganzzahlige MATLAB-Ausgabe näher nachzubilden.
ROUND_TO_UINT8 = True

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
# Vergleichsbild im Datensatz suchen
# ------------------------------------------------------------

# Pfade vereinheitlichen, damit Groß-/Kleinschreibung und
# unterschiedliche Schreibweisen der Trennzeichen nicht stören.
normalized_target_path = os.path.normcase(
    os.path.abspath(
        os.path.normpath(image_path)
    )
)

normalized_dataset_paths = [
    os.path.normcase(
        os.path.abspath(
            os.path.normpath(path)
        )
    )
    for path in file_paths
]

try:
    target_index = normalized_dataset_paths.index(
        normalized_target_path
    )

except ValueError as error:
    same_filename_paths = [
        path
        for path in file_paths
        if os.path.basename(path)
        == os.path.basename(image_path)
    ]

    additional_information = ""

    if same_filename_paths:
        additional_information = (
            "\n\nDateien mit demselben Dateinamen:\n"
            + "\n".join(same_filename_paths)
        )

    raise ValueError(
        "Das Vergleichsbild wurde in dataset.file_paths "
        "nicht gefunden.\n"
        f"Gesuchter Pfad:\n{image_path}"
        f"{additional_information}"
    ) from error


print("\n----------------------------------------")
print("Gefundenes Vergleichsbild")
print("----------------------------------------")
print("Datensatzindex:", target_index)
print("Dateipfad:", file_paths[target_index])


# ------------------------------------------------------------
# Vergleichsbild selbst dekodieren und mit Antialiasing skalieren
#
# image_dataset_from_directory wird oben nur verwendet, um
# dieselbe Klassenreihenfolge und dieselben Dateipfade wie im
# ursprünglichen Testskript zu erhalten.
# ------------------------------------------------------------

selected_image_path = file_paths[target_index]

decode_channels = 1 if channels == 1 else 3

image_bytes = tf.io.read_file(selected_image_path)

image_tensor = tf.io.decode_image(
    image_bytes,
    channels=decode_channels,
    expand_animations=False
)

# decode_image kennt die räumliche Form zunächst nicht statisch.
image_tensor.set_shape([
    None,
    None,
    decode_channels
])

# Explizit bilinear auf die Modelleingabegröße verkleinern.
image_tensor = tf.image.resize(
    image_tensor,
    size=(height, width),
    method=tf.image.ResizeMethod.BILINEAR,
    preserve_aspect_ratio=False,
    antialias=USE_ANTIALIAS
)

if ROUND_TO_UINT8:
    image_tensor = tf.cast(
        tf.clip_by_value(
            tf.round(image_tensor),
            0.0,
            255.0
        ),
        tf.uint8
    )

image_python = image_tensor.numpy()

# Klassenindex aus dem direkten Elternordner bestimmen.
target_class_name = os.path.basename(
    os.path.dirname(selected_image_path)
)

if target_class_name not in class_names:
    raise ValueError(
        "Der Klassenordner des Vergleichsbildes ist nicht "
        "in class_names enthalten.\n"
        f"Klassenordner: {target_class_name}"
    )

label_python = class_names.index(target_class_name)

print("\nResize-Einstellungen:")
print("Bilineare Interpolation: True")
print("Antialiasing:", USE_ANTIALIAS)
print("Rundung auf uint8:", ROUND_TO_UINT8)


print("\n----------------------------------------")
print("Python-Bildeigenschaften")
print("----------------------------------------")

print("Datentyp:", image_python.dtype)
print("Bildgröße:", image_python.shape)
print("Minimum:", float(np.min(image_python)))
print("Maximum:", float(np.max(image_python)))
print("Tatsächlicher Klassenindex:", label_python)
print("Tatsächliche Klasse:", class_names[label_python])

print("\nErster Pixel:")

if image_python.ndim == 3:
    print(image_python[0, 0, :])
else:
    print(image_python[0, 0])

print("\nMittelwert je Kanal:")

if image_python.ndim == 3:
    for channel in range(image_python.shape[2]):
        print(
            f"Kanal {channel + 1}: "
            f"{np.mean(image_python[:, :, channel]):.6f}"
        )
else:
    print(
        f"Kanal 1: {np.mean(image_python):.6f}"
    )


# ------------------------------------------------------------
# Bildgröße kontrollieren
# ------------------------------------------------------------

expected_shape = tuple(model.input_shape[1:])
actual_shape = tuple(image_python.shape)

if actual_shape != expected_shape:
    raise ValueError(
        "Das aus dem Dataset geladene Bild besitzt nicht "
        "die erwartete Form.\n"
        f"Modell erwartet: {expected_shape}\n"
        f"Dataset liefert: {actual_shape}"
    )


# ------------------------------------------------------------
# TensorFlow-Vorhersage
# ------------------------------------------------------------

# Das Bild besitzt bereits die Ausgabe der neuen Resize-Pipeline.
x_python = np.expand_dims(
    image_python,
    axis=0
).astype(np.float32)

scores_tensorflow = model.predict(
    x_python,
    verbose=0
)

if isinstance(scores_tensorflow, (list, tuple)):
    scores_tensorflow = scores_tensorflow[0]

scores_tensorflow = np.asarray(
    scores_tensorflow
).reshape(-1)


print("\n----------------------------------------")
print("TensorFlow-Vorhersage")
print("----------------------------------------")

tensorflow_class_index = int(
    np.argmax(scores_tensorflow)
)

print(
    "Vorhergesagter Klassenindex ab 0:",
    tensorflow_class_index
)

if tensorflow_class_index < len(class_names):
    print(
        "Vorhergesagte Klasse:",
        class_names[tensorflow_class_index]
    )

print("\nTensorFlow-Scores:")
print(scores_tensorflow)


# ------------------------------------------------------------
# MATLAB-Ergebnis laden
# ------------------------------------------------------------

comparison = loadmat(comparison_file)

required_variables = [
    "I_MATLAB",
    "scoresMATLAB"
]

for variable_name in required_variables:
    if variable_name not in comparison:
        raise KeyError(
            f"Die Variable '{variable_name}' fehlt in der "
            "MATLAB-Datei.\n"
            f"Vorhandene Variablen: {list(comparison.keys())}"
        )

image_matlab = np.asarray(
    comparison["I_MATLAB"]
)

scores_matlab = np.asarray(
    comparison["scoresMATLAB"]
).reshape(-1)


print("\n----------------------------------------")
print("MATLAB-Bildeigenschaften")
print("----------------------------------------")

print("Datentyp:", image_matlab.dtype)
print("Bildgröße:", image_matlab.shape)
print("Minimum:", float(np.min(image_matlab)))
print("Maximum:", float(np.max(image_matlab)))


# ------------------------------------------------------------
# Bildgrößen vergleichen
# ------------------------------------------------------------

if image_matlab.shape != image_python.shape:
    raise ValueError(
        "MATLAB und die TensorFlow-Dataset-Pipeline liefern "
        "unterschiedliche Bildgrößen.\n"
        f"MATLAB:     {image_matlab.shape}\n"
        f"TensorFlow: {image_python.shape}\n\n"
        "Falls MATLAB das Bild anders skaliert, muss der "
        "MATLAB-Vergleichscode entsprechend angepasst werden."
    )


# ------------------------------------------------------------
# Pixelwerte vergleichen
# ------------------------------------------------------------

image_matlab_float = image_matlab.astype(np.float32)
image_python_float = image_python.astype(np.float32)

pixel_difference_rgb = np.abs(
    image_matlab_float
    - image_python_float
)

# Zusätzlich Rot-Blau-Vertauschung prüfen
if (
    image_python_float.ndim == 3
    and image_python_float.shape[2] == 3
):
    image_python_bgr = image_python_float[:, :, ::-1]

    pixel_difference_bgr = np.abs(
        image_matlab_float
        - image_python_bgr
    )
else:
    pixel_difference_bgr = None


print("\n----------------------------------------")
print("Pixelvergleich MATLAB gegen Dataset")
print("----------------------------------------")

print(
    "Pixelwerte vollständig identisch:",
    np.array_equal(
        image_matlab_float,
        image_python_float
    )
)

print(
    "Pixelwerte numerisch identisch:",
    np.allclose(
        image_matlab_float,
        image_python_float,
        rtol=0.0,
        atol=1e-6
    )
)

print(
    "Anzahl abweichender Kanalwerte:",
    int(
        np.count_nonzero(
            pixel_difference_rgb > 1e-6
        )
    )
)

print(
    "Maximale Pixelabweichung RGB:",
    float(np.max(pixel_difference_rgb))
)

print(
    "Mittlere Pixelabweichung RGB:",
    float(np.mean(pixel_difference_rgb))
)

# Zusätzlich wird geprüft, wie groß die Abweichung nach einer
# Rundung der Python-Werte auf uint8 bleibt. Dadurch werden reine
# Unterschiede durch Nachkommastellen getrennt betrachtet.
image_python_rounded_uint8 = np.clip(
    np.rint(image_python_float),
    0,
    255
).astype(np.uint8)

pixel_difference_rounded = np.abs(
    image_matlab.astype(np.int16)
    - image_python_rounded_uint8.astype(np.int16)
)

print(
    "\nAnzahl abweichender Kanalwerte nach uint8-Rundung:",
    int(np.count_nonzero(pixel_difference_rounded))
)

print(
    "Maximale Abweichung nach uint8-Rundung:",
    int(np.max(pixel_difference_rounded))
)

print(
    "Mittlere Abweichung nach uint8-Rundung:",
    float(np.mean(pixel_difference_rounded))
)

print(
    "Abweichungen größer als 1:",
    int(np.count_nonzero(pixel_difference_rounded > 1))
)

print(
    "Abweichungen größer als 5:",
    int(np.count_nonzero(pixel_difference_rounded > 5))
)

print(
    "Abweichungen größer als 10:",
    int(np.count_nonzero(pixel_difference_rounded > 10))
)

if pixel_difference_bgr is not None:
    print(
        "\nMaximale Abweichung bei BGR-Vergleich:",
        float(np.max(pixel_difference_bgr))
    )

    print(
        "Mittlere Abweichung bei BGR-Vergleich:",
        float(np.mean(pixel_difference_bgr))
    )


# ------------------------------------------------------------
# Vorhersagen vergleichen
# ------------------------------------------------------------

if len(scores_matlab) != len(scores_tensorflow):
    raise ValueError(
        "Die Anzahl der Modellausgänge unterscheidet sich.\n"
        f"MATLAB:     {len(scores_matlab)}\n"
        f"TensorFlow: {len(scores_tensorflow)}"
    )

score_difference = np.abs(
    scores_matlab
    - scores_tensorflow
)

matlab_class_index = int(
    np.argmax(scores_matlab)
)

tensorflow_class_index = int(
    np.argmax(scores_tensorflow)
)


print("\n----------------------------------------")
print("Vorhersagevergleich")
print("----------------------------------------")

print(
    "MATLAB-Klassenindex ab 0:",
    matlab_class_index
)

print(
    "TensorFlow-Klassenindex ab 0:",
    tensorflow_class_index
)

print(
    "Gleiche vorhergesagte Klasse:",
    matlab_class_index == tensorflow_class_index
)

print(
    "Größte Score-Abweichung:",
    float(np.max(score_difference))
)

print(
    "Mittlere Score-Abweichung:",
    float(np.mean(score_difference))
)


# ------------------------------------------------------------
# Scores tabellarisch ausgeben
# ------------------------------------------------------------

print("\n----------------------------------------")
print("Scores je Ausgang")
print("----------------------------------------")

print(
    f"{'Index':>5} "
    f"{'MATLAB':>14} "
    f"{'TensorFlow':>14} "
    f"{'Abweichung':>14}"
)

for index in range(len(scores_matlab)):
    print(
        f"{index:5d} "
        f"{scores_matlab[index]:14.8f} "
        f"{scores_tensorflow[index]:14.8f} "
        f"{score_difference[index]:14.8f}"
    )