import os

# Muss vor dem TensorFlow-Import stehen
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from scipy.io import loadmat

# Von exportNetworkToTensorFlow erzeugtes Paket
import tf_model


tf.get_logger().setLevel("ERROR")


# ------------------------------------------------------------
# Einstellungen
# ------------------------------------------------------------

# Pfad zu der in MATLAB erzeugten Datei
comparison_file = (
    r"C:\Users\nicoh\OneDrive\Desktop\Matlabcode_Teamprojekt\06_python\TestML_Modell in Phython"
    r"\direktvergleich.mat"
)


# ------------------------------------------------------------
# TensorFlow-Modell laden
# ------------------------------------------------------------

model = tf_model.load_model(
    load_weights=True,
    debug=False
)

print("----------------------------------------")
print("TensorFlow-Modell geladen")
print("----------------------------------------")
print("Eingabe-Shape:", model.input_shape)
print("Ausgabe-Shape:", model.output_shape)


# ------------------------------------------------------------
# MATLAB-Vergleichsdaten laden
# ------------------------------------------------------------

if not os.path.isfile(comparison_file):
    raise FileNotFoundError(
        "Die Datei wurde nicht gefunden:\n"
        f"{comparison_file}"
    )

comparison = loadmat(comparison_file)

required_variables = [
    "XTensorFlow",
    "scoresMATLAB"
]

for variable_name in required_variables:
    if variable_name not in comparison:
        raise KeyError(
            f"Die Variable '{variable_name}' wurde in der "
            "MAT-Datei nicht gefunden.\n"
            f"Vorhandene Variablen: {list(comparison.keys())}"
        )


# ------------------------------------------------------------
# Exakt denselben Eingabetensor aus MATLAB übernehmen
# ------------------------------------------------------------

x_tensorflow = comparison["XTensorFlow"].astype(np.float32)

scores_matlab = np.asarray(
    comparison["scoresMATLAB"],
    dtype=np.float32
).reshape(-1)


# ------------------------------------------------------------
# Eingabeform kontrollieren
# ------------------------------------------------------------

expected_input_shape = model.input_shape

if isinstance(expected_input_shape, list):
    expected_input_shape = expected_input_shape[0]

print("\n----------------------------------------")
print("Eingabetensor")
print("----------------------------------------")
print("Shape aus MAT-Datei:", x_tensorflow.shape)
print("Vom Modell erwartet:", expected_input_shape)
print("Datentyp:", x_tensorflow.dtype)
print("Minimum:", float(np.min(x_tensorflow)))
print("Maximum:", float(np.max(x_tensorflow)))
print("Mittelwert:", float(np.mean(x_tensorflow)))


# Die Batch-Dimension darf im Modell None sein.
expected_without_batch = tuple(expected_input_shape[1:])
actual_without_batch = tuple(x_tensorflow.shape[1:])

if actual_without_batch != expected_without_batch:
    raise ValueError(
        "Die Bilddimensionen stimmen nicht überein.\n"
        f"Modell erwartet: {expected_without_batch}\n"
        f"MAT-Datei enthält: {actual_without_batch}"
    )


# ------------------------------------------------------------
# TensorFlow-Vorhersage
# ------------------------------------------------------------

scores_tensorflow = model.predict(
    x_tensorflow,
    verbose=0
)

# Falls das Modell mehrere Ausgaben besitzt
if isinstance(scores_tensorflow, (list, tuple)):
    scores_tensorflow = scores_tensorflow[0]

scores_tensorflow = np.asarray(
    scores_tensorflow,
    dtype=np.float32
).reshape(-1)


# ------------------------------------------------------------
# Anzahl der Ausgabewerte kontrollieren
# ------------------------------------------------------------

if len(scores_matlab) != len(scores_tensorflow):
    raise ValueError(
        "MATLAB und TensorFlow liefern unterschiedlich viele "
        "Ausgabewerte.\n"
        f"MATLAB: {len(scores_matlab)}\n"
        f"TensorFlow: {len(scores_tensorflow)}"
    )


# ------------------------------------------------------------
# Ergebnisse vergleichen
# ------------------------------------------------------------

difference = np.abs(
    scores_matlab - scores_tensorflow
)

matlab_class_index = int(np.argmax(scores_matlab))
tensorflow_class_index = int(np.argmax(scores_tensorflow))


print("\n----------------------------------------")
print("Vorhersagevergleich")
print("----------------------------------------")

print(
    "MATLAB-Klassenindex ab 0:    ",
    matlab_class_index
)

print(
    "TensorFlow-Klassenindex ab 0:",
    tensorflow_class_index
)

print(
    "Gleicher Klassenindex:       ",
    matlab_class_index == tensorflow_class_index
)

print(
    "\nGrößte absolute Abweichung:",
    float(np.max(difference))
)

print(
    "Mittlere absolute Abweichung:",
    float(np.mean(difference))
)


# ------------------------------------------------------------
# Alle Scores tabellarisch ausgeben
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
        f"{difference[index]:14.8f}"
    )


# ------------------------------------------------------------
# Gesamte Arrays ausgeben
# ------------------------------------------------------------

print("\nMATLAB-Scores:")
print(scores_matlab)

print("\nTensorFlow-Scores:")
print(scores_tensorflow)


# ------------------------------------------------------------
# Einfache Bewertung
# ------------------------------------------------------------

maximum_difference = float(np.max(difference))

print("\n----------------------------------------")
print("Bewertung")
print("----------------------------------------")

if maximum_difference < 1e-5:
    print(
        "Die Ergebnisse sind praktisch identisch. "
        "Der TensorFlow-Export funktioniert sehr wahrscheinlich korrekt."
    )

elif maximum_difference < 1e-3:
    print(
        "Es gibt nur geringe numerische Abweichungen. "
        "Der TensorFlow-Export ist wahrscheinlich korrekt."
    )

elif matlab_class_index == tensorflow_class_index:
    print(
        "Die Scores unterscheiden sich deutlich, aber beide Modelle "
        "wählen dieselbe Klasse. Weitere Bilder sollten geprüft werden."
    )

else:
    print(
        "Die Scores und die vorhergesagte Klasse unterscheiden sich. "
        "Damit liegt wahrscheinlich ein Unterschied zwischen dem "
        "MATLAB-Netz und dem exportierten TensorFlow-Modell vor."
    )