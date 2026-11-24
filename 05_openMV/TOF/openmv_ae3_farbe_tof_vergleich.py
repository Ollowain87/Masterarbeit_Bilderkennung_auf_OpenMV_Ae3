# OpenMV AE3: gemeinsame Farb-/ToF-Maske, Vergleich A/B, Version M11.
# In der OpenMV IDE oeffnen und auf der AE3 starten. Kein ML-Modell noetig.
# A: LAB -> Schliessen -> Lochfuellung -> Flachensuche.
# B: dieselbe bereinigte Farbmaske AND ToF-Maske -> Flachensuche.
# Jede Farbe wird separat verarbeitet. Die Gesamtmasken werden gespeichert.
# Ausschnitte stammen aus dem unmarkierten RGB565-Originalbild.
# Die groben ToF-Grenzen koennen Farbflachen abschneiden und Kanten erzeugen.

import csi
import gc
import image
import os
import time
import tof

# AE3/PAG7936: Verarbeitung in VGA, tatsaechlich 640 x 400 Pixel.
GRID_W = 428
GRID_H = 404
GRID_SHIFT_X = -10
GRID_SHIFT_Y = -11

# LAB-Parameter aus der vorliegenden Farbvorverarbeitung.
THRESHOLDS = (
    (0, 100, 12, 127, -5, 80),
    (15, 55, -5, 8, -27, -14),
    (2, 14, -8, 3, -9, 4),
)
COLOR_NAMES = ("Rot", "Blau", "Dunkelblau")
BOX_COLORS = ((255, 0, 0), (0, 100, 255), (220, 0, 220))
CLOSE_RADIUS = (5, 5, 5)
# 600 stammt aus dem zuletzt vorliegenden LAB-/ML-Skript (22.09.).
# Der aeltere reine Vorverarbeitungstest hatte 1200. Bei Bedarf hier aendern.
# Beide Varianten verwenden immer dieselben Flaechengrenzen.
MIN_BOUNDING_BOX_AREA = (600, 600, 600)
MAX_BOUNDING_BOX_AREA_FRACTION = 0.20
PADDING_FRACTION = 0.15
LOCK_GAIN_AND_WHITEBAL = True

# Startwerte fuer Versuche um 30-50 cm; keine kalibrierten Guetegrenzen.
MAX_DISTANCE_MM = 500
# Anzahl direkt aufeinanderfolgender ToF-Auslesungen PRO KAMERABILD.
# Die Mittelung beginnt bei jedem Kamerabild neu.
AVERAGE_FRAMES = 5
# Werte kleiner 50 mm verwerfen; genau 50 mm wird mitgezaehlt.
# Eine Plausibilitaetsgrenze fuer diesen Versuch, kein Sensor-Zielstatus.
MIN_VALID_DISTANCE_MM = 50
# Mindestens ein gueltiger Nahwert: freigeben. Alle gueltigen Werte fern:
# sofort sperren. Ohne gueltigen Wert: UNBEKANNT und freigeben.
# Mittelwert und Spannweite sind nur Protokollwerte, keine Maskenbedingungen.
# Freigegebene Zonen erhalten 10 Pixel Rand. 0 = harte Rastergrenzen.
MASK_MARGIN_PX = 10

# A: Farb-Ausschnitte, B: Ausschnitte der gemeinsamen Farb-/ToF-Maske.
# Eine Variante als Vollbild anzeigen; beide Varianten werden gespeichert.
# Keine Kacheln, keine Skalierung und keine Python-Pixelschleifen.
DISPLAY_MODE = "B"
SAVE_IMAGES = True
# Pro gespeichertem Bildsatz ein Kamerabild mit ToF-Raster und Zonenwerten.
SAVE_TOF_PREVIEW = True
# Alle ToF-Auslesungen protokollieren, auch zwischen gespeicherten Bildern
# und nach MAX_SAVED_SETS. Bei knappem Speicher endet das Protokoll.
SAVE_TOF_VALUES = True
SAVE_ROOT = "/flash/farbe_tof_vergleich"
SAVE_INTERVAL_MS = 3000
MAX_SAVED_SETS = 20
MIN_FREE_BYTES = 2 * 1024 * 1024
JPEG_QUALITY = 80
SAVE_CROPS = True
TEST_NAME = "gemeinsame_maske"
REFERENCE_MM = None  # Nur Protokollnotiz: z.B. 300 bei gemessenen 300 mm.
LOOP_PAUSE_MS = 200
# M11: Ein Kamerabild, dann fuenf frische ToF-Auslesungen direkt nacheinander.
# Erst danach Farb-/ToF-Masken berechnen, anzeigen und ggf. speichern.
# Mittelwert/Spannweite verwenden nur die Messserie dieses Kamerabildes.
# Alle Rohwerte bleiben im Laufprotokoll und in der CSV des Bildsatzes.
# Werte unter MIN_VALID_DISTANCE_MM gehen nicht in die Statistik ein.
# Nur gueltige Einzelwerte entscheiden: ein Nahwert reicht zur Freigabe.
# Alle gueltigen Werte fern: sofort sperren, auch bei hoher Spannweite.
# Vorschau: RGB565-Vollbild als Bytepuffer kopieren, dann ROIs einzeichnen.
# Ausschnitte: ROI zeilenweise in ein eigenes RGB565-Bild kopieren,
# dieses Bild speichern. Kein roi-Parameter beim JPEG-Speichern.
# Masken: verlustfreies PNG bei voller Aufloesung; PGM als Fehler-Rueckfall.
# tof_raster.jpg: gleiche Aufnahme, Zone/Status/Nahanzahl, Minimum und Mittel.
# Ausfuehrliche Schrittmeldungen abgeschaltet; Meldungen mit "vor" sind
# zusaetzlich auskommentiert. Fehler, Zusammenfassungen und Speicherpfade
# bleiben sichtbar.
DEBUG_TRACE = False
TRACE_FRAME = 0

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)


def trace(message):
    if DEBUG_TRACE:
        if TRACE_FRAME:
            print("[Bild %d] %s" % (TRACE_FRAME, message))
        else:
            print(message)


def trace_done(message, started_us):
    if DEBUG_TRACE:
        elapsed_ms = time.ticks_diff(time.ticks_us(), started_us) / 1000.0
        trace("%s | %.1f ms" % (message, elapsed_ms))


def usable_number(value):
    # Nur numerische Plausibilitaet, KEIN ToF-Zielstatus/Signalstaerke.
    return MIN_VALID_DISTANCE_MM <= value < float("inf")


class ZoneAverage:
    def __init__(self, frames):
        self.frames = frames
        self.reset()

    def reset(self):
        # Zu Beginn jedes Kamerabildes: keine Werte aus dem vorherigen Bild.
        self.history = [[0.0] * 64 for _ in range(self.frames)]
        self.sums = [0.0] * 64
        self.counts = [0] * 64
        self.minimums = [0.0] * 64
        self.read_ok_history = [False] * self.frames
        self.next_slot = 0
        self.filled_slots = 0

    def update(self, depth, read_ok=True):
        slot = self.history[self.next_slot]
        self.read_ok_history[self.next_slot] = read_ok
        means = [0.0] * 64
        spreads = [0.0] * 64
        for zone in range(64):
            if usable_number(slot[zone]):
                self.sums[zone] -= slot[zone]
                self.counts[zone] -= 1
            # Rohwerte einschliesslich 0, negativer Werte, NaN/Inf behalten.
            # Ein Lesefehler liefert keinen Rohwert; sein Slot wird markiert.
            slot[zone] = depth[zone] if read_ok else 0.0
            if usable_number(slot[zone]):
                self.sums[zone] += slot[zone]
                self.counts[zone] += 1
            if self.counts[zone]:
                means[zone] = self.sums[zone] / self.counts[zone]
                low = float("inf")
                high = 0.0
                for old_slot in self.history:
                    old = old_slot[zone]
                    if usable_number(old):
                        low = min(low, old)
                        high = max(high, old)
                spreads[zone] = high - low
                self.minimums[zone] = low
            else:
                self.sums[zone] = 0.0
                self.minimums[zone] = 0.0
        self.next_slot = (self.next_slot + 1) % self.frames
        self.filled_slots = min(self.frames, self.filled_slots + 1)
        return means, spreads

    def raw_window(self, zone):
        # Chronologisch: messwert_1_mm ist der aelteste, der letzte ist
        # der neueste. Noch nicht gefuellte Slots und Lesefehler bleiben leer.
        values = [""] * (self.frames - self.filled_slots)
        first = self.next_slot if self.filled_slots == self.frames else 0
        for offset in range(self.filled_slots):
            index = (first + offset) % self.frames
            values.append(self.history[index][zone]
                          if self.read_ok_history[index] else "")
        return values


class ZoneGate:
    def __init__(self):
        self.blocked = [True] * 64

    def update(self, averager, read_ok):
        states = []

        for zone in range(64):
            valid_count = averager.counts[zone]

            if not read_ok or valid_count == 0:
                state = "UNBEKANNT"
            else:
                near_count = sum(
                    MIN_VALID_DISTANCE_MM <= slot[zone] <= MAX_DISTANCE_MM
                    for slot in averager.history
                )
                far_count = valid_count - near_count

                state = "NAH" if near_count >= far_count else "FERN"

            self.blocked[zone] = state != "NAH"
            states.append(state)

        return states


def clipped_rect(x0, y0, x1, y1, width, height):
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def zone_rect(zone, width, height, grid_x, grid_y, margin=0):
    row, col = zone // 8, zone % 8
    return clipped_rect(
        grid_x + col * GRID_W // 8 - margin,
        grid_y + row * GRID_H // 8 - margin,
        grid_x + (col + 1) * GRID_W // 8 + margin,
        grid_y + (row + 1) * GRID_H // 8 + margin,
        width, height,
    )


def build_tof_mask(mask, blocked, grid_x, grid_y):
    width, height = mask.width(), mask.height()
    # Weiss ausserhalb des Rasters und in unbekannten/nahen Zonen.
    mask.draw_rectangle((0, 0, width, height), color=255, fill=True)
    for zone in range(64):
        if blocked[zone]:
            rect = zone_rect(zone, width, height, grid_x, grid_y)
            if rect is not None:
                mask.draw_rectangle(rect, color=0, fill=True)
    # Erst NACH allen schwarzen Rechtecken die weissen Raender eintragen:
    # ihre Freigabe darf nicht von der Reihenfolge der Zonen abhaengen.
    if MASK_MARGIN_PX:
        for zone in range(64):
            if not blocked[zone]:
                rect = zone_rect(zone, width, height, grid_x, grid_y,
                                 MASK_MARGIN_PX)
                if rect is not None:
                    mask.draw_rectangle(rect, color=255, fill=True)


def expand_roi(rect, width, height):
    x, y, box_w, box_h = rect
    padding = int(PADDING_FRACTION * max(box_w, box_h) + 0.5)
    return clipped_rect(x - padding - 1, y - padding - 1,
                        x + box_w + padding + 1,
                        y + box_h + padding + 1, width, height)


def candidates_from_mask(mask, color_index, variant=""):
    width, height = mask.width(), mask.height()
    max_area = MAX_BOUNDING_BOX_AREA_FRACTION * width * height
    candidates = []
    label = "S[%s %s]" % (COLOR_NAMES[color_index], variant)
    # trace(label + " 1: vor find_blobs")
    step_us = time.ticks_us()
    blobs = mask.find_blobs(
        [(255, 255)], x_stride=1, y_stride=1,
        area_threshold=MIN_BOUNDING_BOX_AREA[color_index],
        pixels_threshold=1, merge=False,
    )
    trace_done(label + " 2: find_blobs fertig", step_us)
    # trace(label + " 3: vor Flaechenfilter und expand_roi")
    for blob in blobs:
        # OpenMV 5.x: Eigenschaft; aeltere API: Methode.
        rect = blob.rect
        if callable(rect):
            rect = rect()
        if rect[2] * rect[3] <= max_area:
            candidates.append((rect, expand_roi(rect, width, height),
                               color_index))
    trace(label + " 4: Ausschnitte fertig: %d" % len(candidates))
    return candidates


def compare_masks(frame, tof_mask, filled, union_a, union_b):
    width, height = frame.width(), frame.height()
    # trace("C0: vor Gesamtmasken clear")
    union_a.clear()
    union_b.clear()
    trace("C0: Gesamtmasken clear fertig")
    candidates_a, candidates_b = [], []
    common_us, a_us, b_us = 0, 0, 0
    for color_index, threshold in enumerate(THRESHOLDS):
        label = "C[%s]" % COLOR_NAMES[color_index]
        # trace(label + " 1: vor binary")
        started = time.ticks_us()
        bitmap = frame.binary([threshold], copy=True, to_bitmap=True)
        trace_done(label + " 2: binary fertig", started)
        radius = CLOSE_RADIUS[color_index]
        if radius:
            # trace(label + " 3: vor close")
            step_us = time.ticks_us()
            bitmap.close(radius)
            trace_done(label + " 4: close fertig", step_us)
        # Dieselbe Lochfuellung wie in der vorhandenen Vorverarbeitung.
        # trace(label + " 5: vor Hilfsbild clear")
        filled.clear()
        # trace(label + " 6: Hilfsbild clear fertig; vor draw_image")
        filled.draw_image(bitmap, x=1, y=1)
        # trace(label + " 7: draw_image fertig; vor flood_fill")
        del bitmap
        step_us = time.ticks_us()
        filled.flood_fill((0, 0), seed_threshold=0.0,
                          floating_threshold=0.0, color=255,
                          invert=True, clear_background=True)
        trace_done(label + " 8: flood_fill fertig", step_us)
        # trace(label + " 9: vor copy ohne Rahmen")
        color_mask = filled.copy(roi=(1, 1, width, height))
        trace(label + " 10: copy fertig")
        common_us += time.ticks_diff(time.ticks_us(), started)

        started = time.ticks_us()
        candidates_a.extend(candidates_from_mask(color_mask, color_index, "A"))
        a_us += time.ticks_diff(time.ticks_us(), started)
        # trace(label + " 11: vor b_or fuer A")
        union_a.b_or(color_mask)
        # trace(label + " 12: b_or A fertig; vor b_and ToF")

        started = time.ticks_us()
        # DER ZUSAETZLICHE SCHRITT: Masken gleicher Groesse und gleichen Formats.
        # Danach kein Schliessen/Lochfuellen mehr: gesperrte Pixel bleiben schwarz.
        color_mask.b_and(tof_mask)
        trace(label + " 13: b_and fertig")
        candidates_b.extend(candidates_from_mask(color_mask, color_index, "B"))
        b_us += time.ticks_diff(time.ticks_us(), started)
        # trace(label + " 14: vor b_or fuer B")
        union_b.b_or(color_mask)
        trace(label + " 15: Farbe fertig")
        del color_mask
    candidates_a.sort(key=lambda item: (item[1][1], item[1][0], item[2]))
    candidates_b.sort(key=lambda item: (item[1][1], item[1][0], item[2]))
    trace("C99: Farb-/ToF-Maskenvergleich fertig")
    # Verarbeitungszeit ohne Kamera, Anzeige und Speicherung.
    # Bei DEBUG_TRACE=True sind Diagnoseausgaben in diesen Zeiten enthalten.
    return candidates_a, candidates_b, common_us + a_us, common_us + b_us


def draw_candidates(target, candidates, x=0, y=0, sx=1.0, sy=1.0):
    for number, candidate in enumerate(candidates):
        rect, roi, color_index = candidate
        x0 = x + int(roi[0] * sx)
        y0 = y + int(roi[1] * sy)
        x1 = x + int((roi[0] + roi[2]) * sx)
        y1 = y + int((roi[1] + roi[3]) * sy)
        target.draw_rectangle((x0, y0, max(1, x1 - x0), max(1, y1 - y0)),
                              color=BOX_COLORS[color_index], thickness=2)
        target.draw_string((x0 + 2, y0 + 2), str(number + 1),
                           color=BOX_COLORS[color_index])


def banner(target, x, y, width, label):
    target.draw_rectangle((x, y, width, 14), color=BLACK, fill=True)
    target.draw_string((x + 3, y + 2), label, color=WHITE)


def copy_rgb565(target, source):
    if (target.width(), target.height()) != (source.width(), source.height()):
        raise ValueError("RGB565-Kopie braucht gleiche Bildgroessen")
    # bytearray() verweist auf den Bildspeicher. Die gesamte Zuweisung erfolgt
    # als Bytepufferkopie, ohne Image[i], Farbkonvertierung oder draw_image().
    source_bytes = source.bytearray()
    target_bytes = target.bytearray()
    expected_bytes = source.width() * source.height() * 2
    if len(source_bytes) != expected_bytes or len(target_bytes) != expected_bytes:
        raise ValueError("RGB565-Kopie braucht zwei unkomprimierte RGB565-Bilder")
    target_bytes[:] = source_bytes


def crop_rgb565(source, roi):
    x, y, crop_w, crop_h = roi
    width, height = source.width(), source.height()
    if (crop_w <= 0 or crop_h <= 0 or x < 0 or y < 0 or
            x + crop_w > width or y + crop_h > height):
        raise ValueError("Ausschnitt liegt ausserhalb des Kamerabildes")
    source_bytes = memoryview(source.bytearray())
    if len(source_bytes) != width * height * 2:
        raise ValueError("Ausschnitt braucht ein unkomprimiertes RGB565-Bild")
    cropped = image.Image(crop_w, crop_h, image.RGB565)
    target_bytes = cropped.bytearray()
    row_bytes = crop_w * 2
    if len(target_bytes) != row_bytes * crop_h:
        raise ValueError("Unerwartete RGB565-Ausschnittpuffergroesse")
    # Pro Bildzeile eine Bytepufferkopie. Die Quelle hat width*2 Byte
    # Zeilenlaenge, der neue Ausschnitt nur crop_w*2 Byte.
    # Keine Pixelschleife, Skalierung oder native draw_image-Operation.
    for row in range(crop_h):
        source_start = ((y + row) * width + x) * 2
        target_start = row * row_bytes
        target_bytes[target_start:target_start + row_bytes] = (
            source_bytes[source_start:source_start + row_bytes]
        )
    return cropped


def make_preview(preview, frame, candidates_a, candidates_b, variant, read_ok):
    if variant not in ("A", "B"):
        raise ValueError("Vorschauvariante muss A oder B sein")
    # trace("V1: vor Vollbildkopie fuer " + variant)
    step_us = time.ticks_us()
    copy_rgb565(preview, frame)
    # trace_done("V2: Vollbildkopie fertig; vor ROI-Rechtecken", step_us)
    selected = candidates_a if variant == "A" else candidates_b
    draw_candidates(preview, selected)
    # trace("V3: ROI-Rechtecke fertig; vor Beschriftung")
    description = "Farbe" if variant == "A" else "Farbe+ToF"
    suffix = "" if read_ok else " | ToF fehlt"
    banner(preview, 0, 0, preview.width(),
           "%s %s | A: %d B: %d%s" % (
               variant, description, len(candidates_a), len(candidates_b), suffix
           ))
    trace("V4: Vollbildvorschau fertig")


def short_distance(value, valid_count):
    if not valid_count:
        return "--"
    # Vier Ziffern passen mit m/a in eine Zone; grosse Werte kennzeichnen.
    if value > 9999:
        return ">9k"
    return str(int(value + 0.5))


def draw_tof_label(target, x, y, text, color=WHITE):
    # OpenMV-Standardschrift: 8 x 10 Pixel. RGB565 braucht RGB-Farbtupel.
    width = min(len(text) * 8 + 2, target.width() - x + 1)
    target.draw_rectangle((x - 1, y - 1, width, 12), color=BLACK, fill=True)
    target.draw_string((x, y), text, color=color)


def make_tof_preview(preview, frame, means, counts, states, averager,
                     grid_x, grid_y, frame_number):
    # Sauberes Original in den vorhandenen RGB565-Vorschaupuffer kopieren.
    # Keine Skalierung, kein RGB-draw_image und kein weiterer Bildpuffer.
    copy_rgb565(preview, frame)
    width, height = frame.width(), frame.height()
    left, right = max(0, grid_x), min(width, grid_x + GRID_W)
    top, bottom = max(0, grid_y), min(height, grid_y + GRID_H)
    for index in range(9):
        x = grid_x + index * GRID_W // 8
        y = grid_y + index * GRID_H // 8
        if 0 <= x < width:
            preview.draw_line((x, top, x, bottom - 1), color=YELLOW)
        if 0 <= y < height:
            preview.draw_line((left, y, right - 1, y), color=YELLOW)

    for zone in range(64):
        rect = zone_rect(zone, width, height, grid_x, grid_y)
        if rect is None:
            continue
        x, y, zone_w, zone_h = rect
        if states[zone] == "NAH":
            code, color = "N", (255, 80, 80)
        elif states[zone] == "FERN":
            code, color = "F", (0, 220, 255)
        else:
            code, color = "U", YELLOW
        near_count = sum(
            MIN_VALID_DISTANCE_MM <= slot[zone] <= MAX_DISTANCE_MM
            for slot in averager.history
        )
        # 03N3/5 = Zone 03, NAH, drei Nahwerte unter fuenf gueltigen Werten.
        # * kennzeichnet zweistellige Anzahlen bei einem groesseren Fenster.
        near_text = str(near_count) if near_count < 10 else "*"
        count_text = str(counts[zone]) if counts[zone] < 10 else "*"
        labels = (
            "%02d%s%s/%s" % (zone, code, near_text, count_text),
            "m" + short_distance(averager.minimums[zone], counts[zone]),
            "a" + short_distance(means[zone], counts[zone]),
        )
        # Drei 10-Pixel-Zeilen passen auch in die oben abgeschnittene Zone.
        preview.draw_rectangle((x + 1, y + 1, zone_w - 2,
                                min(32, zone_h - 2)), color=BLACK, fill=True)
        for line, text in enumerate(labels):
            text_y = y + 2 + line * 10
            if text_y + 10 <= y + zone_h - 1:
                preview.draw_string((x + 2, text_y), text,
                                    color=WHITE if line == 2 else color)

    # Achsen wie zeile_0b/spalte_0b in der CSV: 0 bis 7.
    for row in range(8):
        rect = zone_rect(row * 8, width, height, grid_x, grid_y)
        if rect is not None:
            draw_tof_label(preview, max(1, left - 26),
                           rect[1] + rect[3] // 2 - 5, "R%d" % row)
    for col in range(8):
        rect = zone_rect(col, width, height, grid_x, grid_y)
        if rect is not None:
            draw_tof_label(preview, rect[0] + rect[2] // 2 - 8,
                           min(height - 11, bottom + 1), "C%d" % col)

    legend_x = min(width - 105, right + 5)
    legend = ("Bild %d" % frame_number, "ToF in mm", "Z N n/v",
              "N=nah", "F=fern", "U=unbekannt", "n=Nahwerte",
              "v=gueltige", "m=Minimum", "a=Mittelwert",
              "<50 ignor.", "R/C ab 0")
    for line, text in enumerate(legend):
        draw_tof_label(preview, legend_x, 2 + line * 13, text)
    trace("P ToF: Rastervorschau fertig")


def free_flash_bytes():
    status = os.statvfs("/flash")
    return status[0] * status[3]


def ensure_space():
    if free_flash_bytes() < MIN_FREE_BYTES:
        raise OSError("Weniger als 2 MB frei; Speicherung wird beendet")


def save_mask_lossless(mask, base_path):
    # PNG komprimiert die 0/255-Masken verlustfrei. Kein Resize, keine
    # JPEG-Kompression und keine Aenderung an den laufenden GRAYSCALE-Masken.
    png_path = base_path + ".png"
    try:
        mask.save(png_path)
        with open(png_path, "rb") as file:
            if file.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError("Gespeicherte Datei hat keine PNG-Kennung")
        return png_path
    except (OSError, ValueError, RuntimeError) as error:
        print("PNG-Speicherung fehlgeschlagen; PGM-Rueckfall:", error)
        # Eine angefangene PNG-Datei darf nicht als gueltige Maske erscheinen.
        try:
            os.remove(png_path)
        except OSError:
            pass
        pgm_path = base_path + ".pgm"
        mask.save(pgm_path)
        return pgm_path


def prepare_save_dir(width, height, grid_x, grid_y):
    try:
        ensure_space()
        if "farbe_tof_vergleich" not in os.listdir("/flash"):
            os.mkdir(SAVE_ROOT)
        existing = os.listdir(SAVE_ROOT)
        for run in range(1, 1000):
            name = "lauf_%03d" % run
            if name in existing:
                continue
            folder = SAVE_ROOT + "/" + name
            os.mkdir(folder)
            with open(folder + "/parameter.txt", "w") as file:
                for key, value in (
                    ("Version", "M11"), ("Test", TEST_NAME),
                    ("Referenz_mm", REFERENCE_MM), ("Bild", (width, height)),
                    ("GRID_W_H", (GRID_W, GRID_H)),
                    ("GRID_SHIFT_X_Y", (GRID_SHIFT_X, GRID_SHIFT_Y)),
                    ("GRID_X_Y", (grid_x, grid_y)),
                    ("ToF_vflip_hmirror", (True, True)),
                    ("THRESHOLDS", THRESHOLDS), ("CLOSE_RADIUS", CLOSE_RADIUS),
                    ("MIN_BOUNDING_BOX_AREA", MIN_BOUNDING_BOX_AREA),
                    ("MAX_BOX_AREA_FRACTION", MAX_BOUNDING_BOX_AREA_FRACTION),
                    ("PADDING_FRACTION", PADDING_FRACTION),
                    ("MAX_DISTANCE_MM", MAX_DISTANCE_MM),
                    ("AVERAGE_FRAMES", AVERAGE_FRAMES),
                    ("MIN_VALID_DISTANCE_MM", MIN_VALID_DISTANCE_MM),
                    ("MASK_MARGIN_PX", MASK_MARGIN_PX),
                    ("LOCK_GAIN_AND_WHITEBAL", LOCK_GAIN_AND_WHITEBAL),
                    ("DEBUG_TRACE", DEBUG_TRACE),
                    ("DISPLAY_MODE", DISPLAY_MODE),
                    ("SAVE_IMAGES", SAVE_IMAGES),
                    ("SAVE_TOF_PREVIEW", SAVE_TOF_PREVIEW),
                    ("SAVE_TOF_VALUES", SAVE_TOF_VALUES),
                    ("SAVE_INTERVAL_MS", SAVE_INTERVAL_MS),
                    ("MAX_SAVED_SETS", MAX_SAVED_SETS),
                ):
                    file.write("%s: %s\n" % (key, value))
                file.write("A: LAB -> Schliessen -> Lochfuellung -> Suche\n")
                file.write("B: gleiche Farbmaske AND ToF -> Suche\n")
                file.write("Unbekannt/ausserhalb ToF: freigeben\n")
                file.write("Keine ToF-Zielstatus- oder Signalstaerkedaten\n")
                file.write("Keine ML-Inferenz und keine zusaetzliche Formpruefung\n")
                file.write("JPG: Original, vergleich_A, vergleich_B, tof_raster, Ausschnitte\n")
                file.write("tof_raster: gleiche Aufnahme; Zonen 00-63, R/C ab 0 wie CSV\n")
                file.write("tof_raster Zeile 1: Zone, N/F/U, Nahanzahl/gueltige Anzahl\n")
                file.write("tof_raster Zeile 2: m Minimum der gueltigen Werte in mm\n")
                file.write("tof_raster Zeile 3: a Mittelwert der gueltigen Werte in mm\n")
                file.write("tof_raster: -- ohne gueltigen Wert; >9k oberhalb 9999 mm; * Anzahl >=10\n")
                file.write("Nahanzahl ist keine Signalstaerke und kein Sensor-Zielstatus\n")
                file.write("Masken: PNG, verlustfrei in voller Aufloesung; PGM bei PNG-Fehler\n")
                file.write("Diagnoseausgaben beeinflussen Verarbeitungszeiten\n")
                file.write("Vorschau: A oder B als Vollbild mit ROI-Rechtecken\n")
                file.write("Bildkopien: Bytepuffer, ohne Skalierung und ohne Pixelschleifen\n")
                file.write("Ausschnitte: eigenes ROI-Bild, dann JPEG ohne roi-Parameter\n")
                file.write("Pro Kamerabild AVERAGE_FRAMES ToF-Auslesungen direkt nacheinander\n")
                file.write("Mittelwert/Spannweite nur aus der Messserie dieses Kamerabildes\n")
                file.write("Kamerabild -> frame.flush -> ToF-Serie -> Masken -> Anzeige/Speicherung\n")
                file.write("Keine Mittelungswerte aus vorherigen Kamerabildern\n")
                file.write("tof_zonen.csv: messwert_1_mm bis messwert_N_mm, aeltester bis neuester\n")
                file.write("Einzelwerte bleiben unveraendert; Mittelwert/Spannweite nur aus endlichen Werten >= %s mm\n" % MIN_VALID_DISTANCE_MM)
                file.write("Verbleibende gueltige Werte verwenden; ohne gueltigen Wert UNBEKANNT\n")
                file.write("Mindestens ein gueltiger Einzelwert <= MAX_DISTANCE_MM: NAH, freigeben\n")
                file.write("Alle gueltigen Einzelwerte > MAX_DISTANCE_MM: FERN, sofort sperren\n")
                file.write("Keine Spannweitenpruefung, keine Hysterese, keine Bestaetigung durch weitere Kamerabilder\n")
                file.write("Mittelwert/Spannweite nur Protokollwerte; Zonenentscheidung aus kleinstem gueltigen Einzelwert\n")
                file.write("Leere Einzelwertfelder: noch kein Wert oder ToF-Lesefehler\n")
                file.write("tof_einzelwerte.csv: jede Auslesung; kamerabild entspricht messwerte.csv\n")
                file.write("tof_nr: Auslesung 1 bis AVERAGE_FRAMES innerhalb dieses Kamerabildes\n")
                file.write("tof_laufzeit_ms: Zeit seit Schleifenstart, direkt nach ToF-Auslesung\n")
                file.write("tof_ok=1: 64 Werte gelesen, keine Aussage ueber Zielstatus/Signalstaerke\n")
                file.write("tof_ok=0: keine vollstaendige Auslesung, alle 64 Zonenfelder leer\n")
                file.write("messwerte.csv tof_ok: alle Auslesungen der Bildserie erfolgreich\n")
                file.write("ToF-Protokoll laeuft nach dem Bildlimit bis zum Stop oder Speicherlimit weiter\n")
            with open(folder + "/messwerte.csv", "w") as file:
                file.write("satz;kamerabild;laufzeit_ms;tof_ok;A_anzahl;B_anzahl;"
                           "gesperrte_zonen;unbekannt_unruhig;tof_lesen_ms;"
                           "tof_maske_ms;A_verarbeitung_ms;B_verarbeitung_ms\n")
            if SAVE_TOF_VALUES:
                with open(folder + "/tof_einzelwerte.csv", "w") as file:
                    columns = ["kamerabild", "tof_nr", "tof_laufzeit_ms", "tof_ok"]
                    columns += ["zone_%02d_mm" % zone for zone in range(64)]
                    file.write(";".join(columns) + "\n")
            return folder
        print("Alle Laufnummern belegt; Livevergleich ohne Speicherung")
    except OSError as error:
        print("Speicherung nicht verfuegbar:", error)
    return None


def append_tof_reading(folder, frame_number, reading_number, elapsed, raw, read_ok):
    # Eine Zeile pro ToF-Auslesung: fuenf Zeilen je Kamerabild.
    # Bei Lesefehlern keine kuenstlichen Nullmessungen protokollieren.
    ensure_space()
    values = [frame_number, reading_number, elapsed, int(read_ok)]
    values += list(raw) if read_ok else [""] * 64
    with open(folder + "/tof_einzelwerte.csv", "a") as file:
        file.write(";".join(str(value) for value in values) + "\n")


def read_tof_batch(averager, frame_number, started, log_folder):
    # Eine eigenstaendige Messserie fuer genau dieses Kamerabild.
    averager.reset()
    tof_us = 0
    readings = []
    for reading_number in range(1, averager.frames + 1):
        # read_depth wartet laut OpenMV auf einen neuen Sensorframe.
        # Daher keine kuenstliche Pause und keine Kamerabild-/Maskenarbeit
        # zwischen den fuenf Auslesungen.
        sample_started = time.ticks_us()
        sample_ok = True
        try:
            raw, dmin, dmax = tof.read_depth(
                vflip=True, hmirror=True, timeout=250
            )
            if len(raw) != 64:
                raise RuntimeError("ToF liefert nicht 64 Werte")
        except RuntimeError as error:
            print("ToF-Lesefehler | Bild %d | Auslesung %d/%d:" % (
                frame_number, reading_number, averager.frames
            ), error)
            raw = [0.0] * 64
            sample_ok = False
        tof_us += time.ticks_diff(time.ticks_us(), sample_started)
        elapsed = time.ticks_diff(time.ticks_ms(), started)
        readings.append((raw, sample_ok, elapsed))
    # Erst NACH den fuenf Auslesungen mitteln und protokollieren.
    # So verzoegert die CSV-Speicherung die Messserie nicht.
    averaging_started = time.ticks_us()
    for raw, sample_ok, elapsed in readings:
        means, spreads = averager.update(raw, sample_ok)
    averaging_us = time.ticks_diff(time.ticks_us(), averaging_started)
    if log_folder:
        try:
            for index, (raw, sample_ok, elapsed) in enumerate(readings):
                append_tof_reading(log_folder, frame_number, index + 1,
                                   elapsed, raw, sample_ok)
        except (OSError, MemoryError) as error:
            print("ToF-Protokoll beendet; Livevergleich laeuft weiter:", error)
            log_folder = None
    # Bei Lesefehlern in der Serie vorsichtshalber freigeben.
    read_ok = all(averager.read_ok_history)
    return (readings[-1][0], means, spreads, read_ok, tof_us,
            averaging_us, log_folder)


def save_set(folder, number, frame_number, elapsed, frame, preview,
             tof_mask, union_a, union_b, candidates_a, candidates_b,
             raw, means, counts, spreads, states, gate, read_ok,
             grid_x, grid_y, timing, averager):
    # trace("P1: vor Speicherplatzpruefung und Aufnahmeordner")
    ensure_space()
    name = "bild_%03d" % number
    destination = folder + "/" + name
    os.mkdir(destination)
    # Alle Bilder und Ausschnitte beziehen sich auf dasselbe saubere Kamerabild.
    # trace("P2: vor original.jpg speichern")
    frame.save(destination + "/original.jpg", quality=JPEG_QUALITY)
    for variant in ("A", "B"):
        ensure_space()
        # trace("P3: vor Vorschau und Speicherung vergleich_" + variant + ".jpg")
        make_preview(preview, frame, candidates_a, candidates_b, variant, read_ok)
        preview.save(destination + "/vergleich_" + variant + ".jpg",
                     quality=JPEG_QUALITY)
        trace("P3: vergleich_" + variant + ".jpg fertig")
    if SAVE_TOF_PREVIEW:
        ensure_space()
        # trace("P ToF: vor tof_raster.jpg zeichnen und speichern")
        make_tof_preview(preview, frame, means, counts, states, averager,
                         grid_x, grid_y, frame_number)
        preview.save(destination + "/tof_raster.jpg", quality=JPEG_QUALITY)
        trace("P ToF: tof_raster.jpg fertig")
    # trace("P4: beide Vergleichsbilder fertig; vor Masken speichern")
    for label, mask in (("tof_maske", tof_mask), ("farbmaske", union_a),
                        ("gemeinsame_maske", union_b)):
        ensure_space()
        # trace("P Maske: vor " + label + ".png")
        step_us = time.ticks_us()
        filename = save_mask_lossless(mask, destination + "/" + label)
        trace_done("P Maske: %s fertig | %d Byte" % (
            filename.rsplit("/", 1)[-1], os.stat(filename)[6]
        ), step_us)
    # trace("P5: vor tof_zonen.csv schreiben")
    with open(destination + "/tof_zonen.csv", "w") as file:
        header = "zone;zeile_0b;spalte_0b;roh_mm;mittel_mm;anzahl;"
        header += "spannweite_mm;zustand;gesperrt;x;y;breite;hoehe"
        for index in range(averager.frames):
            header += ";messwert_%d_mm" % (index + 1)
        file.write(header + "\n")
        for zone in range(64):
            rect = zone_rect(zone, frame.width(), frame.height(), grid_x, grid_y)
            rect_values = ["", "", "", ""] if rect is None else list(rect)
            values = [zone, zone // 8, zone % 8, raw[zone], means[zone],
                      counts[zone], spreads[zone], states[zone],
                      int(gate.blocked[zone])] + rect_values
            values += averager.raw_window(zone)
            file.write(";".join(str(value) for value in values) + "\n")
    # trace("P6: ToF-CSV fertig; vor Ausschnitten und Ausschnitt-CSV")
    with open(destination + "/ausschnitte.csv", "w") as file:
        file.write("variante;nummer;farbe;box_x;box_y;box_breite;box_hoehe;"
                   "roi_x;roi_y;roi_breite;roi_hoehe;datei\n")
        for variant, candidates in (("A", candidates_a), ("B", candidates_b)):
            for index, candidate in enumerate(candidates):
                rect, roi, color_index = candidate
                filename = ""
                if SAVE_CROPS:
                    ensure_space()
                    filename = "%s_%02d_%s.jpg" % (
                        variant, index + 1, COLOR_NAMES[color_index]
                    )
                    # trace("P Ausschnitt: vor " + filename)
                    cropped = crop_rgb565(frame, roi)
                    trace("P Ausschnitt: %s | ROI %s | Bild %d x %d" % (
                        filename, str(roi), cropped.width(), cropped.height()
                    ))
                    cropped.save(destination + "/" + filename,
                                 quality=JPEG_QUALITY)
                    del cropped
                    trace("P Ausschnitt: " + filename + " fertig")
                values = [variant, index + 1, COLOR_NAMES[color_index]]
                values += list(rect) + list(roi) + [filename]
                file.write(";".join(str(value) for value in values) + "\n")
    # trace("P7: Ausschnitte fertig; vor Messwerte und fertig.txt")
    # Der CSV-Spaltenname unbekannt_unruhig bleibt fuer alte Auswertungen.
    # M11 erzeugt keine UNRUHIG-Zonen mehr; hier nur UNBEKANNT zaehlen.
    unknown = sum(state == "UNBEKANNT" for state in states)
    values = [name, frame_number, elapsed, int(read_ok), len(candidates_a),
              len(candidates_b), sum(gate.blocked), unknown] + list(timing)
    with open(folder + "/messwerte.csv", "a") as file:
        file.write(";".join(str(value) for value in values) + "\n")
    # Nur Serien mit diesem Marker sind vollstaendig gespeichert.
    with open(destination + "/fertig.txt", "w") as file:
        file.write("Aufnahme vollstaendig\n")
    trace("P8: Bildsatz vollstaendig gespeichert")
    return name


def main():
    global TRACE_FRAME
    TRACE_FRAME = 0
    if AVERAGE_FRAMES < 1:
        raise ValueError("AVERAGE_FRAMES muss mindestens 1 sein")
    if (MAX_DISTANCE_MM <= 0 or
            MASK_MARGIN_PX < 0 or MIN_VALID_DISTANCE_MM <= 0 or
            MIN_VALID_DISTANCE_MM > MAX_DISTANCE_MM):
        raise ValueError("ToF-Maskenparameter pruefen")
    if DISPLAY_MODE not in ("A", "B"):
        raise ValueError("DISPLAY_MODE muss A oder B sein")
    print("AE3 gemeinsame Farb-/ToF-Maske M11 | fuenf ToF-Auslesungen pro Bild")
    cam = csi.CSI()
    cam.reset()
    cam.pixformat(csi.RGB565)
    cam.framesize(csi.VGA)
    time.sleep_ms(1000)
    # print("1: vor erstem Kamerabild")
    frame = cam.snapshot()
    frame.flush()
    width, height = frame.width(), frame.height()
    print("Tatsaechliche Bildgroesse:", width, "x", height)
    if (width, height) != (640, 400):
        raise ValueError("Dieses ToF-Raster ist fuer 640 x 400 abgestimmt")
    if LOCK_GAIN_AND_WHITEBAL:
        time.sleep_ms(1000)
        frame = cam.snapshot()
        frame.flush()
        cam.auto_gain(False)
        cam.auto_whitebal(False)
    grid_x = (width - GRID_W) // 2 + GRID_SHIFT_X
    grid_y = (height - GRID_H) // 2 + GRID_SHIFT_Y
    # print("2: Kamera bereit; vor tof.init")
    tof.init()
    time.sleep_ms(1000)
    print("3: ToF bereit | Raster:", grid_x, grid_y, GRID_W, GRID_H)
    print("Suchbereich <=", MAX_DISTANCE_MM, "mm | Rand:", MASK_MARGIN_PX,
          "Pixel | Mittelung:", AVERAGE_FRAMES)
    print("Weiss = freigegeben; Schwarz = durch ToF ausgeschlossen")
    print("A/B verwenden dasselbe Kamerabild; kein ML-Modell noetig")
    print("Vollbildanzeige:", DISPLAY_MODE, "| gespeichert werden A und B")
    print("Pro Kamerabild:", AVERAGE_FRAMES,
          "ToF-Auslesungen direkt nacheinander; Mittelung nur fuer dieses Bild")
    print("Werte unter", MIN_VALID_DISTANCE_MM,
          "mm verwerfen; ohne gueltigen Nahwert sofort FERN")
    if SAVE_IMAGES and SAVE_TOF_PREVIEW:
        print("Je Bildsatz: tof_raster.jpg mit Zonenwerten und R/C ab 0")

    averager, gate = ZoneAverage(AVERAGE_FRAMES), ZoneGate()
    tof_mask = image.Image(width, height, image.GRAYSCALE)
    union_a = image.Image(width, height, image.GRAYSCALE)
    union_b = image.Image(width, height, image.GRAYSCALE)
    filled = image.Image(width + 2, height + 2, image.GRAYSCALE)
    preview = image.Image(width, height, image.RGB565)
    folder = (prepare_save_dir(width, height, grid_x, grid_y)
              if SAVE_IMAGES or SAVE_TOF_VALUES else None)
    if folder:
        print("Speicherordner:", folder)
    log_folder = folder if SAVE_TOF_VALUES else None
    if log_folder:
        print("Alle ToF-Auslesungen:", log_folder + "/tof_einzelwerte.csv")
    if not SAVE_IMAGES:
        folder = None
    saved, frame_number = 0, 0
    started = time.ticks_ms()
    last_save = last_report = started
    print("4: Hauptschleife startet")
    while True:
        frame_number += 1
        TRACE_FRAME = frame_number
        # trace("L1: vor cam.snapshot")
        step_us = time.ticks_us()
        frame = cam.snapshot()
        # trace_done("L2: cam.snapshot fertig; vor fruehem frame.flush", step_us)
        # Bewaehrte Reihenfolge fuer den AE3-Kamera-/ToF-Dauerbetrieb.
        frame.flush()
        # trace("L3: fruehes frame.flush fertig; vor tof.read_depth")
        # Fuer das aktuelle Kamerabild fuenfmal hintereinander auslesen.
        # read_tof_batch setzt dabei alle Mittelungswerte zurueck.
        raw, means, spreads, read_ok, tof_us, averaging_us, log_folder = read_tof_batch(
            averager, frame_number, started, log_folder
        )
        mask_started = time.ticks_us()
        # trace("L5: Mittelung fertig; vor ZoneGate")
        states = gate.update(averager, read_ok)
        # trace("L6: ZoneGate fertig; vor ToF-Maske zeichnen")
        build_tof_mask(tof_mask, gate.blocked, grid_x, grid_y)
        trace("L7: ToF-Maske fertig")
        mask_us = averaging_us + time.ticks_diff(time.ticks_us(), mask_started)
        # trace("L8: vor Farb-/ToF-Maskenvergleich")
        candidates_a, candidates_b, a_us, b_us = compare_masks(
            frame, tof_mask, filled, union_a, union_b
        )
        timing = (tof_us / 1000.0, mask_us / 1000.0,
                  a_us / 1000.0, (b_us + tof_us + mask_us) / 1000.0)
        trace("L9: Maskenvergleich fertig")
        now = time.ticks_ms()
        trace("L10: Speicherintervall pruefen")
        if folder and time.ticks_diff(now, last_save) >= SAVE_INTERVAL_MS:
            # trace("L11: vor Speicherung")
            try:
                name = save_set(folder, saved + 1, frame_number,
                                time.ticks_diff(now, started), frame, preview,
                                tof_mask, union_a, union_b, candidates_a,
                                candidates_b, raw, means, averager.counts,
                                spreads, states, gate, read_ok, grid_x, grid_y,
                                timing, averager)
                saved += 1
                last_save = time.ticks_ms()
                print("Gespeichert:", folder + "/" + name,
                      "(%d/%d)" % (saved, MAX_SAVED_SETS))
                if saved >= MAX_SAVED_SETS:
                    print("Speicherserie beendet; Livevergleich laeuft weiter")
                    folder = None
            except (OSError, MemoryError) as error:
                print("Speicherung beendet; Livevergleich laeuft weiter:", error)
                print("Eine unvollstaendige Aufnahme hat keinen fertig.txt-Marker")
                folder = None
        # trace("L12: vor Vollbildvorschau")
        make_preview(preview, frame, candidates_a, candidates_b,
                     DISPLAY_MODE, read_ok)
        # Erst jetzt darf das saubere Kamerabild durch die Anzeige ersetzt werden.
        # trace("L13: Anzeigemodus fertig; vor Vorschau in Kamerabild kopieren")
        copy_rgb565(frame, preview)
        # trace("L14: Kopieren fertig; vor abschliessendem frame.flush")
        frame.flush()
        trace("L15: abschliessendes frame.flush fertig")
        if frame_number == 1 or time.ticks_diff(now, last_report) >= 1000:
            print("Bild", frame_number, "| A:", len(candidates_a),
                  "| B:", len(candidates_b), "| gesperrte Zonen:",
                  sum(gate.blocked), "| ToF-Auslesungen:", AVERAGE_FRAMES,
                  "| ToF ms:", timing[0], "| A/B Verarbeitung ms:",
                  timing[2], timing[3])
            last_report = now
        # trace("L16: vor gc.collect")
        gc.collect()
        # trace("L17: Speicherbereinigung fertig; vor Schleifenpause")
        time.sleep_ms(LOOP_PAUSE_MS)
        trace("L18: Pause beendet; Bilddurchlauf fertig")


if __name__ == "__main__":
    main()
