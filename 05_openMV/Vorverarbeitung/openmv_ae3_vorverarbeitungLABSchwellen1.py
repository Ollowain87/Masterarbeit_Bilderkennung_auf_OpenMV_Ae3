import gc
import image
import os


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
# mindestgröße der Boundingbox ohne Vergrößerung
MIN_BOUNDING_BOX_AREA = (600, 600, 600)
MAX_BOUNDING_BOX_AREA_FRACTION = 0.20
# Vergrößerung des Auschnitts um diesen Prozentwert
PADDING_FRACTION = 0.15

DRAW_BOXES = True

# ---------------------------------------------------
# SPEICHERN
# ---------------------------------------------------

OUTPUT_DIR = "/flash/bbox_debug"
CSV_FILE = OUTPUT_DIR + "/bounding_boxes.csv"

# JPEG-Qualitaet
JPEG_QUALITY = 90


def init_output():
    """Ausgabeordner und CSV-Datei anlegen."""

    try:
        os.mkdir(OUTPUT_DIR)
    except OSError:
        # Ordner existiert wahrscheinlich bereits.
        pass

    # CSV bei jedem Neustart neu anlegen.
    with open(CSV_FILE, "w") as f:
        f.write(
            "Bild;Box;Farbe;X;Y;Breite;Hoehe;Flaeche\n"
        )


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


def expand_roi(rect, image_width, image_height):
    """Box um 15 % der laengeren Seite erweitern."""

    x, y, width, height = rect

    padding = int(
        PADDING_FRACTION * max(width, height) + 0.5
    )

    x1 = max(0, x - padding - 1)
    y1 = max(0, y - padding - 1)

    x2 = min(
        image_width,
        x + width + padding + 1
    )

    y2 = min(
        image_height,
        y + height + padding + 1
    )

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1
    )


def find_candidates(frame):
    """
    Liefert:
    [
        ((x, y, breite, hoehe), farbindex),
        ...
    ]
    """

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
        image.GRAYSCALE
    )

    for color_index, threshold in enumerate(THRESHOLDS):

        mask = frame.binary(
            [threshold],
            copy=True,
            to_bitmap=True
        )

        radius = CLOSE_RADIUS[color_index]

        if radius > 0:
            mask.close(radius)

        # Lochfuellung
        filled_mask.clear()

        filled_mask.draw_image(
            mask,
            1,
            1
        )

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

            # Wichtig fuer AE3:
            # rect ist ein Tuple und keine Funktion.
            x, y, box_width, box_height = blob.rect

            if box_width * box_height > max_area:
                continue

            # Kuenstlichen Rand entfernen.
            rect = (
                x - 1,
                y - 1,
                box_width,
                box_height
            )

            roi = expand_roi(
                rect,
                width,
                height
            )

            candidates.append(
                (roi, color_index)
            )

    # Oben -> unten, dann links -> rechts
    candidates.sort(
        key=lambda item: (
            item[0][1],
            item[0][0]
        )
    )

    return candidates


def save_results(frame, candidates, frame_number):
    """
    Speichert:
    1. komplettes Bild mit Boundingboxen
    2. Boundingbox-Daten in CSV
    """

    image_name = "bild_%06d.jpg" % frame_number
    image_path = OUTPUT_DIR + "/" + image_name

    # -------------------------------------------
    # CSV schreiben
    # -------------------------------------------

    with open(CSV_FILE, "a") as f:

        # Falls keine Box gefunden wurde,
        # trotzdem das Bild protokollieren.
        if len(candidates) == 0:

            f.write(
                "%s;0;Keine;;;;;\n"
                % image_name
            )

        else:

            for box_number, item in enumerate(candidates):

                roi, color_index = item

                x, y, width, height = roi

                area = width * height

                f.write(
                    "%s;%d;%s;%d;%d;%d;%d;%d\n"
                    % (
                        image_name,
                        box_number + 1,
                        COLOR_NAMES[color_index],
                        x,
                        y,
                        width,
                        height,
                        area
                    )
                )

    # -------------------------------------------
    # Bild MIT eingezeichneten Boxen speichern
    # -------------------------------------------

    frame.save(
        image_path,
        quality=JPEG_QUALITY
    )


def main():

    camera = init_camera()

    init_output()

    frame_number = 0

    while True:

        frame_number += 1

        frame = camera.snapshot()

        candidates = find_candidates(frame)

        # ----------------------------------------
        # Ausschnitte erzeugen
        # ----------------------------------------

        for roi, color_index in candidates:

            crop = frame.copy(roi=roi)

            # Spaeter:
            # ML-Inferenz mit crop

            del crop

        # ----------------------------------------
        # Boundingboxen zeichnen
        # ----------------------------------------

        if DRAW_BOXES:

            for roi, color_index in candidates:

                frame.draw_rectangle(
                    roi,
                    color=BOX_COLORS[color_index],
                    thickness=2
                )

        # ----------------------------------------
        # Bild + Boxgroessen speichern
        # ----------------------------------------

        save_results(
            frame,
            candidates,
            frame_number
        )

        gc.collect()


if __name__ == "__main__":
    main()
