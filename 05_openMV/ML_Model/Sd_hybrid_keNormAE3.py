import ml
import gc
import image
import os
import time
# -----------------------------
# Einstellungen
# -----------------------------
input_folder = "/flash/03_13Kat_Alle Schilder_Ae3"
output_folder = "/flash/bilderopenmv"
model_path = "/flash/models/sign-classifier_hybrid_npu.tflite"
csv_path = output_folder + "/ergebnisse.csv"

imgSize = 48

# Reihenfolge muss den 13 Modellausgängen entsprechen
labels = [
    "Achtung_Gefahrenstelle",
    "BilderOhneSchilder",
    "Durchfahrt_Verboten_Einbahnstrasse",
    "Fussgaengerueberweg",
    "Geschwindigkeitsbegrenzung_100",
    "Geschwindigkeitsbegrenzung_30",
    "Geschwindigkeitsbegrenzung_50",
    "Geschwindigkeitsbegrenzung_70",
    "Geschwindigkeitsempfehlung_30",
    "Geschwindigkeitsempfehlung_90",
    "Stoppschild",
    "Verkehrsberuhigter_Bereich",
    "Vorfahrt_Achten"
]


# -----------------------------
# Ausgabeordner erstellen
# -----------------------------
def create_output_folder():
    try:
        os.mkdir(output_folder)
    except OSError:
        pass


# -----------------------------
# CSV-Datei neu anlegen
# -----------------------------
def create_csv_file():
    create_output_folder()

    f = open(csv_path, "w")
    f.write(
        "Dateiname;RichtigeKlasse;"
        "ErkannteKlasse;RichtigErkannt\n"
    )
    f.close()


# -----------------------------
# CSV-Zeile ergänzen
# -----------------------------
def append_csv_line(
    datei,
    richtige_klasse,
    erkannte_klasse,
    richtig_erkannt
):
    f = open(csv_path, "a")

    f.write("{};{};{};{}\n".format(
        datei,
        richtige_klasse,
        erkannte_klasse,
        richtig_erkannt
    ))

    f.close()


# -----------------------------
# Bild mit Area-Averaging
# auf 48 x 48 verkleinern
# -----------------------------
def resize_to_48(src):
    dst = image.Image(
        imgSize,
        imgSize,
        image.RGB565
    )

    x_scale = imgSize / src.width()
    y_scale = imgSize / src.height()

    dst.draw_image(
        src,
        0,
        0,
        x_scale=x_scale,
        y_scale=y_scale,
        hint=image.AREA
    )

    return dst


# -----------------------------
# Bilder aus Unterordnern laden
# -----------------------------
bilder = []
print("Aktuelles Verzeichnis:", os.getcwd())
print("Inhalt von /:", os.listdir("/"))
print("Inhalt von /flash:", os.listdir("/flash"))
print("Gesuchter Ordner:", input_folder)

for richtige_klasse in os.listdir(input_folder):
    klassen_ordner = input_folder + "/" + richtige_klasse

    try:
        print("Gesuchter KlassenOrdner:", klassen_ordner)
        dateien = os.listdir(klassen_ordner)
    except OSError:
        continue

    for datei in dateien:
        datei_klein = datei.lower()

        if (
            datei_klein.endswith(".jpg")
            or datei_klein.endswith(".jpeg")
            or datei_klein.endswith(".png")
            or datei_klein.endswith(".bmp")
        ):
            bilder.append((
                klassen_ordner + "/" + datei,
                datei,
                richtige_klasse
            ))


# -----------------------------
# Modell laden
# -----------------------------
try:
    model = ml.Model(
        model_path)
except Exception as e:
    print("Fehler beim Laden des Modells:", e)
    raise e


# -----------------------------
# CSV vorbereiten
# -----------------------------
create_csv_file()

print(
    "Dateiname;RichtigeKlasse;"
    "ErkannteKlasse;RichtigErkannt"
)


# -----------------------------
# Alle Bilder verarbeiten
# -----------------------------
startzeit = time.ticks_ms()
for pfad_zum_bild, datei, richtige_klasse in bilder:

    try:
        # Bild von SD-Karte laden
        sd_img = image.Image(pfad_zum_bild)

        # In RGB565 umwandeln
        img = sd_img.to_rgb565(copy=True)

        # Auf 48 x 48 verkleinern
        small = resize_to_48(img)

        # Bild in Eingabetensor umwandeln
        image_array = small.to_ndarray(
            dtype="f"
        ).reshape((1, imgSize, imgSize, 3))

        start_inferenz = time.ticks_us()

        output = model.predict([image_array])

        inferenz_us = time.ticks_diff(
            time.ticks_us(),
            start_inferenz
        )

        print(
            "Inferenzzeit: {:.3f} ms".format(
                inferenz_us / 1000
            )
        )

        # Alle Ausgabewerte eindimensional machen
        scores = output[0].flatten()

        if len(scores) != len(labels):
            raise Exception(
                "Modell hat {} Ausgaenge, aber {} Labels".format(
                    len(scores),
                    len(labels)
                )
            )

        # Klasse mit dem höchsten Score suchen
        erkannte_klasse_index = 0
        hoechster_score = scores[0]

        for i in range(1, len(scores)):
            if scores[i] > hoechster_score:
                hoechster_score = scores[i]
                erkannte_klasse_index = i

        erkannte_klasse = labels[erkannte_klasse_index]

        if erkannte_klasse == richtige_klasse:
            richtig_erkannt = "JA"
        else:
            richtig_erkannt = "NEIN"

        # Konsolenausgabe
        print("{};{};{};{}".format(
            datei,
            richtige_klasse,
            erkannte_klasse,
            richtig_erkannt
        ))

        # CSV-Ausgabe
        append_csv_line(
            datei,
            richtige_klasse,
            erkannte_klasse,
            richtig_erkannt
        )

    except Exception as e:
        print("{};{};FEHLER;{}".format(
            datei,
            richtige_klasse,
            e
        ))

        # Nur bei einem Fehler eine Fehlerzeile schreiben
        append_csv_line(
            datei,
            richtige_klasse,
            "FEHLER",
            "NEIN"
        )

    gc.collect()

laufzeit_ms = time.ticks_diff(time.ticks_ms(), startzeit)
laufzeit_s = laufzeit_ms / 1000

print("")
print("Auswertung abgeschlossen.")
print("CSV:", csv_path)
print("Gesamtlaufzeit: {:.2f} Sekunden".format(laufzeit_s))
