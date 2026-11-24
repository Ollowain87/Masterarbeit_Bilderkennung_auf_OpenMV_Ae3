import ml
import gc
import sensor, image, time
import os

za = 0

# Ausgabeordner
output_folder = "/sdcard/bilderopenmv"

def create_output_folder():
    try:
        os.mkdir(output_folder)
        print("Ordner erstellt:", output_folder)
    except OSError:
        pass

def savePictureWithPrediction(filenamePrefix, filenameSuffix, imageToSave):
    create_output_folder()

    tm = time.localtime()

    # Zeitstempel: HH_MM_SS
    formattedTimestamp = "{:02d}_{:02d}_{:02d}".format(tm[3], tm[4], tm[5])

    filename = "{}/{}____{}__{}.bmp".format(
        output_folder,
        filenamePrefix,
        formattedTimestamp,
        filenameSuffix
    )

    imageToSave.save(filename)
    print("Bild gespeichert:", filename)


sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)

pfad = "/sdcard/Quantisierungsbilder"
dateien = os.listdir(pfad)
bilder = [f for f in dateien if f.endswith(".jpg") or f.endswith(".png")]

fb = sensor.snapshot()

print("trying to load")
print("Free mem before loading model:", gc.mem_free())

model = None
try:
    model = ml.Model("/sdcard/models/sign-classifier_hybrid.tflite", load_to_fb=False)
except Exception as e:
    print(e)

print("loaded")
print("Free mem after loading model:", gc.mem_free())

labels = ['keinSchild', 'Schild']

print("Model input shape:", model.input_shape)
print("Model input dtype:", model.input_dtype)
print("Model output shape:", model.output_shape)
print("Model output dtype:", model.output_dtype)


for datei in bilder:
    pfad_zum_bild = pfad + "/" + datei

    sd_img = image.Image(pfad_zum_bild)
    img = sd_img.to_rgb565(copy_to_fb=True)

    # Originalbild sichern, bevor Kreise eingezeichnet werden
    img_original = img.copy()

    print("Verarbeite:", datei)

    circles = img.find_circles(threshold=3500,
                               x_margin=10,
                               y_margin=10,
                               r_margin=10,
                               r_min=2,
                               r_max=45,
                               r_step=2)

    if not circles:
        continue

    #print("Suche Kreis… Gefundene Kreise:", len(circles))
    #print("Kreis gefunden! Zeichne ein.")

    za = 0

    for c in circles:
        za += 1
        print("Kreis", za)

        x, y, r = c.x(), c.y(), c.r()

        # Nur zur Anzeige auf img zeichnen
        img.draw_circle(x, y, r, color=(255, 0, 0))

        # ROI berechnen
        roi = (x - 24, y - 24, 48, 48)

        if (roi[0] < 0 or roi[1] < 0 or
            roi[0] + roi[2] > img.width() or
            roi[1] + roi[3] > img.height()):
            print("ROI außerhalb:", roi)
            continue

        # Ausschnitt aus dem unveränderten Originalbild nehmen
        small = img_original.copy(roi=roi)

        # Für nicht vollständig quantisiertes Modell:
        # float32, Wertebereich 0.0 bis 255.0
        image_array = small.to_ndarray(dtype="f").reshape((1, 48, 48, 3))

        try:
            output = model.predict([image_array])
            scores = output[0][0][0][0]

            print("Klassifikationsergebnis: [{:.7f}, {:.7f}]".format(scores[0], scores[1]))

            if scores[0] > scores[1]:
                print("Erkannt:", labels[0], "Wert:", "{:.7f}".format(scores[0]))
            else:
                print("Erkannt:", labels[1], "Wert:", "{:.7f}".format(scores[1]))

            # Wahrscheinlichkeiten in Prozent
            klasse0_prozent = scores[0] * 100
            klasse1_prozent = scores[1] * 100

            # Dateiname mit Klassenvorhersagen
            filenamePrefix = "keinSchild_{:.1f}____Schild_{:.1f}".format(
                scores[0] * 100,
                scores[1] * 100
            )

            # Kreisnummer + ursprünglicher Dateiname als Zusatz
            filenameSuffix = "{:03d}_{}".format(za, datei)

            # Bildausschnitt speichern
            savePictureWithPrediction(filenamePrefix, filenameSuffix, small)

        except Exception as e:
            print("Fehler bei model.predict:", e)

        time.sleep_ms(2000)
