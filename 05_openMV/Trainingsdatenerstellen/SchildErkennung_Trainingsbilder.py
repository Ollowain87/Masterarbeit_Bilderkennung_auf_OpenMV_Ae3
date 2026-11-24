# OpenMV AE3: LAB-Farbsegmentierung + 48x48-ML-Klassifikation
# Ausgelegt fuer OpenMV Firmware 5.0.0.

import gc
import image
import ml
import os
import time


# ---------------------------------------------------
# FARBVORVERARBEITUNG
# ---------------------------------------------------

# Reihenfolge: L_min, L_max, a_min, a_max, b_min, b_max.
THRESHOLDS = (
    (0, 100, 12, 127, -5, 80),   # Rot
    (15, 55, -5, 8, -27, -14),   # Blau
    (2, 14, -8, 3, -9, 4),       # Dunkelblau
)

COLOR_NAMES = ("Rot", "Blau", "Dunkelblau")
BOX_COLORS = ((255, 0, 0), (0, 100, 255), (220, 0, 220))

FRAME_SIZE = "VGA"

CLOSE_RADIUS = (5, 5, 5)

# Mindestflaeche der Bounding Box VOR der Vergroesserung.
MIN_BOUNDING_BOX_AREA = (400, 400, 400)
MAX_BOUNDING_BOX_AREA_FRACTION = 0.10

# Vergroesserung um 15 % der laengeren Boxseite auf allen vier Seiten.
PADDING_FRACTION = 0.15

DRAW_BOXES = True
DRAW_LABELS = True


# ---------------------------------------------------
# ML-MODELL
# ---------------------------------------------------

MODEL_PATH = "/flash/models/sign-classifier_hybrid_npu.tflite"
MODEL_INPUT_SIZE = 48

# Reihenfolge muss exakt den 13 Modellausgaengen entsprechen.
LABELS = (
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
    "Vorfahrt_Achten",
)


# ---------------------------------------------------
# SPEICHERN
# ---------------------------------------------------

OUTPUT_ROOT = "/flash/training_crops"
JPEG_QUALITY = 90


def init_output():
    """Fuer jeden Start einen neuen Ausgabeordner anlegen."""
    try:
        os.mkdir(OUTPUT_ROOT)
    except OSError:
        # Ordner existiert wahrscheinlich bereits.
        pass

    # Ein eigener Sitzungsordner verhindert, dass vorhandene
    # Trainingsbilder bei einem Neustart ueberschrieben werden.
    existing_entries = os.listdir(OUTPUT_ROOT)
    session_number = 1

    while True:
        session_name = "session_%03d" % session_number

        if session_name not in existing_entries:
            break

        session_number += 1

    output_dir = OUTPUT_ROOT + "/" + session_name
    os.mkdir(output_dir)

    csv_file = output_dir + "/training_crops.csv"

    with open(csv_file, "w") as file:
        file.write(
            "Ausschnitt;Bild;Box;Farbe;X;Y;Breite;Hoehe;Flaeche;"
            "Klassenindex_0basiert;ErkannteKlasse;Modellwert;Inferenz_ms\n"
        )

    print("Trainingsbilder werden gespeichert in:", output_dir)
    return output_dir, csv_file


def init_camera():
    """RGB565-Kamera starten und Gain/Weissabgleich einregeln lassen."""
    try:
        import csi
    except ImportError:
        import sensor

        sensor.reset()
        sensor.set_pixformat(sensor.RGB565)
        sensor.set_framesize(getattr(sensor, FRAME_SIZE))
        sensor.skip_frames(time=2000)
        sensor.set_auto_gain(False)
        sensor.set_auto_whitebal(False)
        sensor.skip_frames(time=200)
        return sensor

    camera = csi.CSI()
    camera.reset()
    camera.pixformat(csi.RGB565)
    camera.framesize(getattr(csi, FRAME_SIZE))
    camera.snapshot(time=2000)
    camera.auto_gain(False)
    camera.auto_whitebal(False)
    camera.snapshot(time=200)
    return camera


def load_model():
    """Modell genau einmal laden und die erwartete Eingabegroesse pruefen."""
    try:
        model = ml.Model(MODEL_PATH)
    except Exception as error:
        print("Fehler beim Laden des Modells:", error)
        raise error

    expected_shape = (1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3)

    if (
        len(model.input_shape) != 1
        or tuple(model.input_shape[0]) != expected_shape
    ):
        raise ValueError(
            "Modell erwartet {}, benoetigt wird {}".format(
                model.input_shape,
                expected_shape,
            )
        )

    print("Modell geladen:", MODEL_PATH)
    print("Modelleingang:", model.input_shape, model.input_dtype)
    print("Modellausgang:", model.output_shape, model.output_dtype)
    return model


def expand_roi(rect, image_width, image_height):
    """Box um 15 % der laengeren Seite erweitern und am Bildrand begrenzen."""
    x, y, width, height = rect

    padding = int(
        PADDING_FRACTION * max(width, height) + 0.5
    )

    x1 = max(0, x - padding - 1)
    y1 = max(0, y - padding - 1)
    x2 = min(image_width, x + width + padding + 1)
    y2 = min(image_height, y + height + padding + 1)

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    )


def find_candidates(frame):
    """Liefert [((x, y, breite, hoehe), farbindex), ...]."""
    if frame.format() != image.RGB565:
        raise ValueError(
            "find_candidates erwartet ein RGB565-Farbbild"
        )

    width = frame.width()
    height = frame.height()

    max_area = (
        MAX_BOUNDING_BOX_AREA_FRACTION
        * width
        * height
    )

    candidates = []

    # Schwarzer Aussenrand fuer flood_fill.
    filled_mask = image.Image(
        width + 2,
        height + 2,
        image.GRAYSCALE,
    )

    for color_index, threshold in enumerate(THRESHOLDS):
        mask = frame.binary(
            [threshold],
            copy=True,
            to_bitmap=True,
        )

        radius = CLOSE_RADIUS[color_index]

        if radius > 0:
            mask.close(radius)

        # Lochfuellung entsprechend MATLAB imfill(..., 'holes').
        filled_mask.clear()
        filled_mask.draw_image(mask, x=1, y=1)
        del mask

        filled_mask.flood_fill(
            (0, 0),
            seed_threshold=0.0,
            floating_threshold=0.0,
            color=255,
            invert=True,
            clear_background=True,
        )

        blobs = filled_mask.find_blobs(
            [(255, 255)],
            roi=(1, 1, width, height),
            x_stride=1,
            y_stride=1,
            area_threshold=MIN_BOUNDING_BOX_AREA[color_index],
            pixels_threshold=1,
            merge=False,
        )

        for blob in blobs:
            x, y, box_width, box_height = blob.rect

            if box_width * box_height > max_area:
                continue

            # Den kuenstlichen Aussenrand entfernen.
            rect = (
                x - 1,
                y - 1,
                box_width,
                box_height,
            )

            roi = expand_roi(rect, width, height)
            candidates.append((roi, color_index))

    # Oben nach unten, danach links nach rechts.
    candidates.sort(
        key=lambda item: (
            item[0][1],
            item[0][0],
        )
    )

    del filled_mask
    return candidates


def resize_to_48(source):
    """RGB565-Ausschnitt mit Area-Resampling auf exakt 48x48 bringen."""
    resized = image.Image(
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE,
        image.RGB565,
    )

    x_scale = MODEL_INPUT_SIZE / source.width()
    y_scale = MODEL_INPUT_SIZE / source.height()

    resized.draw_image(
        source,
        x=0,
        y=0,
        x_scale=x_scale,
        y_scale=y_scale,
        hint=image.AREA,
    )

    return resized


def classify_crop(model, crop):
    """Ausschnitt klassifizieren, ohne die Pixelwerte zu normalisieren."""
    small = resize_to_48(crop)

    # WICHTIG: keine Division durch 255 und keine Skalierung auf -1 bis 1.
    # dtype='f' liefert dem Modell RGB-Werte im Bereich 0.0 bis 255.0.
    input_tensor = small.to_ndarray(
        dtype="f"
    ).reshape((1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3))

    start_us = time.ticks_us()
    output = model.predict([input_tensor])
    inference_us = time.ticks_diff(time.ticks_us(), start_us)

    scores = output[0].flatten()

    if len(scores) != len(LABELS):
        raise ValueError(
            "Modell hat {} Ausgaenge, aber {} Labels".format(
                len(scores),
                len(LABELS),
            )
        )

    best_index = 0
    best_score = float(scores[0])

    for index in range(1, len(scores)):
        current_score = float(scores[index])

        if current_score > best_score:
            best_score = current_score
            best_index = index

    label = LABELS[best_index]
    inference_ms = inference_us / 1000.0

    # Grosse temporaere Objekte vor dem naechsten Kandidaten freigeben.
    del scores
    del output
    del input_tensor
    del small

    return best_index, label, best_score, inference_ms


def draw_detections(frame, detections):
    """Boxen und ML-Ergebnis erst nach allen Inferenzen einzeichnen."""
    if not DRAW_BOXES:
        return

    for box_number, detection in enumerate(detections):
        (
            roi,
            color_index,
            class_index,
            label,
            score,
            inference_ms,
        ) = detection

        frame.draw_rectangle(
            roi,
            color=BOX_COLORS[color_index],
            thickness=2,
        )

        if DRAW_LABELS:
            x, y, width, height = roi
            label_text = "{}:{} {:.3f}".format(
                box_number + 1,
                label,
                score,
            )

            # 8 Pixel je Zeichen bei scale=1 grob beruecksichtigen.
            text_width = len(label_text) * 8
            text_x = min(x, max(0, frame.width() - text_width))

            if y >= 12:
                text_y = y - 12
            else:
                text_y = min(frame.height() - 10, y + height + 2)

            frame.draw_string(
                (text_x, text_y),
                label_text,
                color=BOX_COLORS[color_index],
                scale=1,
            )


def save_clean_crop(crop, output_dir, frame_number, box_number):
    """Unveraenderten ROI ohne Box, Text oder sonstige Markierung speichern."""
    crop_name = "bild_%06d_box_%02d.jpg" % (
        frame_number,
        box_number,
    )
    crop_path = output_dir + "/" + crop_name

    crop.save(
        crop_path,
        quality=JPEG_QUALITY,
    )

    return crop_name


def append_crop_metadata(
    csv_file,
    crop_name,
    frame_number,
    box_number,
    detection,
):
    """Position und aktuelle Modellvorhersage separat in die CSV schreiben."""
    (
        roi,
        color_index,
        class_index,
        label,
        score,
        inference_ms,
    ) = detection

    x, y, width, height = roi
    area = width * height

    with open(csv_file, "a") as file:
        file.write(
            "%s;%d;%d;%s;%d;%d;%d;%d;%d;%d;%s;%.6f;%.3f\n"
            % (
                crop_name,
                frame_number,
                box_number,
                COLOR_NAMES[color_index],
                x,
                y,
                width,
                height,
                area,
                class_index,
                label,
                score,
                inference_ms,
            )
        )


def main():
    model = load_model()
    camera = init_camera()
    output_dir, csv_file = init_output()

    frame_number = 0

    while True:
        frame_number += 1
        frame = camera.snapshot()
        candidates = find_candidates(frame)
        detections = []

        # Zuerst alle Originalausschnitte klassifizieren.
        for box_number, candidate in enumerate(candidates):
            roi, color_index = candidate

            # Die Kopie entsteht aus dem noch voellig unmarkierten Gesamtbild.
            crop = frame.copy(roi=roi)

            # Sofort die Originalgroesse speichern. Das Offline-Training kann
            # danach mit exakt derselben Resize-Methode wie bisher arbeiten.
            crop_name = save_clean_crop(
                crop,
                output_dir,
                frame_number,
                box_number + 1,
            )

            (
                class_index,
                label,
                score,
                inference_ms,
            ) = classify_crop(model, crop)

            detection = (
                roi,
                color_index,
                class_index,
                label,
                score,
                inference_ms,
            )
            detections.append(detection)

            append_crop_metadata(
                csv_file,
                crop_name,
                frame_number,
                box_number + 1,
                detection,
            )

            print(
                "Bild {} Box {} | {} -> {} | Wert {:.6f} | {:.3f} ms".format(
                    frame_number,
                    box_number + 1,
                    COLOR_NAMES[color_index],
                    label,
                    score,
                    inference_ms,
                )
            )

            del crop
            gc.collect()

        # Erst jetzt zeichnen. Dadurch sind weder die gespeicherten Ausschnitte
        # noch die Modellbilder mit Boxen oder Text beschriftet.
        draw_detections(frame, detections)

        # Das markierte Gesamtbild bleibt nur im Framebuffer fuer die
        # Live-Anzeige im OpenMV-IDE. Es wird absichtlich nicht gespeichert.

        del candidates
        del detections
        gc.collect()


if __name__ == "__main__":
    main()
