# OpenMV AE3: gemeinsame Farb-/ToF-Maske, Vergleich A/B mit ML, M13-GC.
# In der OpenMV IDE oeffnen und auf der AE3 starten. MODEL_PATH pruefen.
# Grundlage: angehaengtes M13-Skript; manuelle GC einmal je Kamerabild.
# A: LAB -> Schliessen -> Lochfuellung -> Flachensuche.
# B: dieselbe bereinigte Farbmaske AND ToF-Maske -> Flachensuche.
# Jede Farbe wird separat verarbeitet. Die Gesamtmasken werden gespeichert.
# Ausschnitte stammen aus dem unmarkierten RGB565-Originalbild.
# Die groben ToF-Grenzen koennen Farbflachen abschneiden und Kanten erzeugen.

import csi
import gc
import image
import ml
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
    (15, 48, -6, 8, -27, -2),
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

# ML wie im vorhandenen LAB-/ML-Skript: 48 x 48 RGB, Area-Resampling,
# Float-RGB-Werte 0..255, KEINE zusaetzliche Normalisierung.
# Das Modell wird einmal geladen; A und B werden getrennt klassifiziert.
# Auch identische ROIs werden zweimal bewertet, damit die Zeiten vergleichbar
# bleiben. Keine Score-Schwelle und kein Entfernen von BilderOhneSchilder.
MODEL_PATH = "/flash/models/sign-classifier_hybrid_npu.tflite"
MODEL_INPUT_SIZE = 48
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
SAVE_ML_RESULTS = True
# ML-Zeiten intern in Mikrosekunden, in saemtlichen CSVs in Millisekunden.
# "gc" bleibt fuer bestehende Auswertungen erhalten und ist pro ROI 0.
# Die manuelle Bereinigung je Kamerabild steht separat in bild_gc_ms.
# Automatische GC bleibt aktiv; ihre Zeit liegt ggf. im ausloesenden Schritt.
ML_TIME_KEYS = (
    "ausschnitt", "resize", "tensor", "inferenz", "auswertung", "gc",
    "gesamt",
)

# Startwerte fuer Versuche um 30-50 cm; keine kalibrierten Guetegrenzen.
MAX_DISTANCE_MM = 500
# Anzahl direkt aufeinanderfolgender ToF-Auslesungen PRO KAMERABILD.
# Die Mittelung beginnt bei jedem Kamerabild neu.
AVERAGE_FRAMES = 5
# Werte kleiner 50 mm verwerfen; genau 50 mm wird mitgezaehlt.
# Eine Plausibilitaetsgrenze fuer diesen Versuch, kein Sensor-Zielstatus.
MIN_VALID_DISTANCE_MM = 50
# Mindestens so viele gueltige Nah- wie Fernwerte: freigeben.
# Mehr Fernwerte: sofort sperren. Ohne gueltigen Wert: UNBEKANNT und sperren.
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
# Gesamtbudget am Bildende fuer manuelle GC, Zeitprotokoll und Restpause.
# Dauern GC und Protokoll laenger, entfaellt die zusaetzliche Schlafpause.
# M11: Ein Kamerabild, dann fuenf frische ToF-Auslesungen direkt nacheinander.
# Erst danach Farb-/ToF-Masken berechnen, anzeigen und ggf. speichern.
# Mittelwert/Spannweite verwenden nur die Messserie dieses Kamerabildes.
# Alle Rohwerte bleiben im Laufprotokoll und in der CSV des Bildsatzes.
# Werte unter MIN_VALID_DISTANCE_MM gehen nicht in die Statistik ein.
# Nur gueltige Einzelwerte entscheiden: Nahanzahl >= Fernanzahl gibt frei.
# Mehr Fernwerte: sofort sperren, auch bei hoher Spannweite.
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

# Zeitmessung ohne Einzelschritt-Ausgaben im Terminal.
# zeitmessung.csv: eine Zeile je Kamerabild, auch ohne gespeicherten Bildsatz.
# bild_gc_ms: gemeinsame GC am Bildende, als zusaetzliche letzte Spalte.
# zeitmessung_farben.csv: drei Zeilen je Kamerabild (Rot/Blau/Dunkelblau).
SAVE_TIMINGS = True
COLOR_TIME_KEYS = (
    "binary", "close", "hilfsbild_clear", "hilfsbild_zeichnen",
    "lochfuellung", "maskenkopie", "farbe_gemeinsam",
    "A_flachensuche", "A_roi", "A_union", "A_zweig",
    "B_and", "B_flachensuche", "B_roi", "B_union", "B_zweig",
)
FRAME_TIME_KEYS = (
    "kamera", "fruehes_flush", "tof_lesen", "tof_mittelung",
    "tof_zonegate", "tof_maske_zeichnen", "tof_maske", "tof_protokoll",
    "farbmasken_clear", "A_clear", "B_clear",
    "binary", "close", "hilfsbild_clear", "hilfsbild_zeichnen",
    "lochfuellung", "maskenkopie", "A_flachensuche", "A_roi", "A_union",
    "B_and", "B_flachensuche", "B_roi", "B_union", "A_sort", "B_sort",
    "farbe_gemeinsam", "A_nur_farbe", "B_ohne_tof", "B_mit_tof",
    "AB_vergleich_wall", "bildspeicherung", "live_vorschau",
    "ablauf_bis_vorschau",
)
# Alte Vorverarbeitungsfelder bleiben unveraendert. Die neuen Gesamtfelder
# addieren jeweils die eigene ML-Verarbeitung zu A bzw. B inklusive ToF.
# A/B-Gesamtzeiten enthalten keine gemeinsam am Bildende ausgefuehrte GC.
FRAME_TIME_KEYS += tuple(
    variant + "_ml_" + key
    for variant in ("A", "B") for key in ML_TIME_KEYS
) + ("A_mit_ml", "B_mit_tof_ml", "AB_ml_wall", "ml_protokoll")

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


def candidates_from_mask(mask, color_index, variant="", timing_row=None):
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
    if timing_row is not None:
        timing_row[variant + "_flachensuche"] = time.ticks_diff(
            time.ticks_us(), step_us
        )
    trace_done(label + " 2: find_blobs fertig", step_us)
    # trace(label + " 3: vor Flaechenfilter und expand_roi")
    roi_started = time.ticks_us()
    for blob in blobs:
        # OpenMV 5.x: Eigenschaft; aeltere API: Methode.
        rect = blob.rect
        if callable(rect):
            rect = rect()
        if rect[2] * rect[3] <= max_area:
            candidates.append((rect, expand_roi(rect, width, height),
                               color_index))
    if timing_row is not None:
        timing_row[variant + "_roi"] = time.ticks_diff(
            time.ticks_us(), roi_started
        )
    trace(label + " 4: Ausschnitte fertig: %d" % len(candidates))
    return candidates


def record_duration(timings, key, started_us):
    # Alle intern gespeicherten Dauern sind Mikrosekunden; CSV in Millisekunden.
    elapsed = time.ticks_diff(time.ticks_us(), started_us)
    timings[key] = timings.get(key, 0) + elapsed
    return elapsed


def compare_masks(frame, tof_mask, filled, union_a, union_b, timings=None):
    if timings is None:
        timings = {}
    comparison_started = time.ticks_us()
    color_rows = []
    width, height = frame.width(), frame.height()
    # trace("C0: vor Gesamtmasken clear")
    step_us = time.ticks_us()
    union_a.clear()
    clear_a_us = record_duration(timings, "A_clear", step_us)
    step_us = time.ticks_us()
    union_b.clear()
    clear_b_us = record_duration(timings, "B_clear", step_us)
    timings["farbmasken_clear"] = clear_a_us + clear_b_us
    trace("C0: Gesamtmasken clear fertig")
    candidates_a, candidates_b = [], []
    common_us, a_us, b_us = 0, clear_a_us, clear_b_us
    for color_index, threshold in enumerate(THRESHOLDS):
        row = {}
        color_rows.append(row)
        label = "C[%s]" % COLOR_NAMES[color_index]
        # trace(label + " 1: vor binary")
        started = time.ticks_us()
        step_us = time.ticks_us()
        bitmap = frame.binary([threshold], copy=True, to_bitmap=True)
        record_duration(row, "binary", step_us)
        trace_done(label + " 2: binary fertig", started)
        radius = CLOSE_RADIUS[color_index]
        if radius:
            # trace(label + " 3: vor close")
            step_us = time.ticks_us()
            bitmap.close(radius)
            record_duration(row, "close", step_us)
            trace_done(label + " 4: close fertig", step_us)
        # Dieselbe Lochfuellung wie in der vorhandenen Vorverarbeitung.
        # trace(label + " 5: vor Hilfsbild clear")
        step_us = time.ticks_us()
        filled.clear()
        record_duration(row, "hilfsbild_clear", step_us)
        # trace(label + " 6: Hilfsbild clear fertig; vor draw_image")
        step_us = time.ticks_us()
        filled.draw_image(bitmap, x=1, y=1)
        record_duration(row, "hilfsbild_zeichnen", step_us)
        # trace(label + " 7: draw_image fertig; vor flood_fill")
        del bitmap
        step_us = time.ticks_us()
        filled.flood_fill((0, 0), seed_threshold=0.0,
                          floating_threshold=0.0, color=255,
                          invert=True, clear_background=True)
        record_duration(row, "lochfuellung", step_us)
        trace_done(label + " 8: flood_fill fertig", step_us)
        # trace(label + " 9: vor copy ohne Rahmen")
        step_us = time.ticks_us()
        color_mask = filled.copy(roi=(1, 1, width, height))
        record_duration(row, "maskenkopie", step_us)
        trace(label + " 10: copy fertig")
        common_us += record_duration(row, "farbe_gemeinsam", started)

        started = time.ticks_us()
        found_a = candidates_from_mask(color_mask, color_index, "A", row)
        candidates_a.extend(found_a)
        row["A_anzahl"] = len(found_a)
        # trace(label + " 11: vor b_or fuer A")
        step_us = time.ticks_us()
        union_a.b_or(color_mask)
        record_duration(row, "A_union", step_us)
        a_us += record_duration(row, "A_zweig", started)
        # trace(label + " 12: b_or A fertig; vor b_and ToF")

        started = time.ticks_us()
        # DER ZUSAETZLICHE SCHRITT: Masken gleicher Groesse und gleichen Formats.
        # Danach kein Schliessen/Lochfuellen mehr: gesperrte Pixel bleiben schwarz.
        step_us = time.ticks_us()
        color_mask.b_and(tof_mask)
        record_duration(row, "B_and", step_us)
        trace(label + " 13: b_and fertig")
        found_b = candidates_from_mask(color_mask, color_index, "B", row)
        candidates_b.extend(found_b)
        row["B_anzahl"] = len(found_b)
        # trace(label + " 14: vor b_or fuer B")
        step_us = time.ticks_us()
        union_b.b_or(color_mask)
        record_duration(row, "B_union", step_us)
        b_us += record_duration(row, "B_zweig", started)
        trace(label + " 15: Farbe fertig")
        del color_mask
    step_us = time.ticks_us()
    candidates_a.sort(key=lambda item: (item[1][1], item[1][0], item[2]))
    a_us += record_duration(timings, "A_sort", step_us)
    step_us = time.ticks_us()
    candidates_b.sort(key=lambda item: (item[1][1], item[1][0], item[2]))
    b_us += record_duration(timings, "B_sort", step_us)
    for key in COLOR_TIME_KEYS:
        timings[key] = sum(row.get(key, 0) for row in color_rows)
    timings["farben"] = color_rows
    timings["farbe_gemeinsam"] = common_us
    timings["A_nur_farbe"] = common_us + a_us
    timings["B_ohne_tof"] = common_us + b_us
    record_duration(timings, "AB_vergleich_wall", comparison_started)
    trace("C99: Farb-/ToF-Maskenvergleich fertig")
    # A/B enthalten den gemeinsamen Farbanteil, eigenes Clear, Union und Sort.
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


def load_model():
    """Ein Modell fuer beide Varianten; Form und Klassenanzahl pruefen."""
    model = ml.Model(MODEL_PATH)
    expected = (1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3)
    if len(model.input_shape) != 1 or tuple(model.input_shape[0]) != expected:
        raise ValueError("Modelleingang %s, erwartet %s" % (
            str(model.input_shape), str(expected)
        ))
    if len(model.output_shape) != 1:
        raise ValueError("Ein Modellausgang mit %d Klassen erwartet" % len(LABELS))
    output_count = 1
    for dimension in model.output_shape[0]:
        output_count *= dimension
    if output_count != len(LABELS):
        raise ValueError("Modell liefert %d Werte, LABELS enthaelt %d Klassen" % (
            output_count, len(LABELS)
        ))
    print("ML-Modell geladen:", MODEL_PATH)
    print("ML-Eingang:", model.input_shape, model.input_dtype)
    print("ML-Ausgang:", model.output_shape, model.output_dtype)
    return model


def resize_to_48(source):
    """Wie im LAB-/ML-Skript, beide Achsen auf genau 48 Pixel skalieren."""
    resized = image.Image(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, image.RGB565)
    resized.draw_image(
        source, x=0, y=0,
        x_scale=MODEL_INPUT_SIZE / source.width(),
        y_scale=MODEL_INPUT_SIZE / source.height(),
        hint=image.AREA,
    )
    return resized


def classify_candidates(model, frame, candidates, variant, timings):
    """Unmarkierte Original-ROIs bewerten, ohne Kandidaten zu filtern."""
    results = []
    for key in ML_TIME_KEYS:
        timings[variant + "_ml_" + key] = 0
    if not candidates:
        return results
    started = time.ticks_us()
    for rect, roi, color_index in candidates:
        roi_started = time.ticks_us()
        result = {}
        step_us = time.ticks_us()
        # Eigener RGB565-Puffer, dieselbe ROI wie bei der JPEG-Speicherung.
        crop = crop_rgb565(frame, roi)
        result["ausschnitt_us"] = time.ticks_diff(time.ticks_us(), step_us)
        step_us = time.ticks_us()
        small = resize_to_48(crop)
        result["resize_us"] = time.ticks_diff(time.ticks_us(), step_us)
        step_us = time.ticks_us()
        # Genau wie vorher: Float-RGB 0..255, NICHT durch 255 teilen.
        input_tensor = small.to_ndarray(dtype="f").reshape(
            (1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3)
        )
        result["tensor_us"] = time.ticks_diff(time.ticks_us(), step_us)
        step_us = time.ticks_us()
        output = model.predict([input_tensor])
        result["inferenz_us"] = time.ticks_diff(time.ticks_us(), step_us)
        step_us = time.ticks_us()
        flat = output[0].flatten()
        if len(flat) != len(LABELS):
            raise ValueError("Unerwartete ML-Klassenanzahl: %d" % len(flat))
        scores = tuple(float(value) for value in flat)
        best_index = 0
        for index in range(1, len(scores)):
            if scores[index] > scores[best_index]:
                best_index = index
        result["class_index"] = best_index
        result["label"] = LABELS[best_index]
        result["score"] = scores[best_index]
        result["scores"] = scores
        result["auswertung_us"] = time.ticks_diff(time.ticks_us(), step_us)
        # Keine grossen temporaeren Bild-/Tensorpuffer pro ROI behalten.
        del flat, output, input_tensor, small, crop
        # Ein manueller GC folgt erst nach dem gesamten Kamerabild.
        result["gc_us"] = 0
        result["gesamt_us"] = time.ticks_diff(time.ticks_us(), roi_started)
        results.append(result)
        for key in ML_TIME_KEYS:
            if key != "gesamt":
                timings[variant + "_ml_" + key] += result[key + "_us"]
    # Gesamtzeit schliesst die Schleife ein; ohne manuelle GC/Anzeige/CSV/JPEG.
    timings[variant + "_ml_gesamt"] = time.ticks_diff(time.ticks_us(), started)
    return results


def ml_result_header():
    header = ["klasse_index_0b", "erkannte_klasse", "modellwert"]
    header += ["ml_" + key + "_ms" for key in ML_TIME_KEYS]
    header += ["modellwert_%02d" % index for index in range(len(LABELS))]
    return header


def ml_result_values(result):
    if result is None:
        return [""] * len(ml_result_header())
    values = [result["class_index"], result["label"], "%.6f" % result["score"]]
    values += ["%.3f" % (result[key + "_us"] / 1000.0)
               for key in ML_TIME_KEYS]
    values += ["%.6f" % value for value in result["scores"]]
    return values


def create_ml_log(folder):
    header = ["kamerabild", "bildsatz", "laufzeit_ms", "variante", "nummer",
              "farbe", "status", "box_x", "box_y", "box_breite", "box_hoehe",
              "roi_x", "roi_y", "roi_breite", "roi_hoehe"]
    header += ml_result_header()
    with open(folder + "/ml_ergebnisse.csv", "w") as file:
        file.write(";".join(header) + "\n")


def append_ml_results(folder, frame_number, saved_name, elapsed,
                      candidates_a, candidates_b, results_a, results_b):
    """Alle Kandidaten beider Varianten, auch ohne gespeicherten Bildsatz."""
    ensure_space()
    with open(folder + "/ml_ergebnisse.csv", "a") as file:
        for variant, candidates, results in (
            ("A", candidates_a, results_a), ("B", candidates_b, results_b)
        ):
            if len(candidates) != len(results):
                raise ValueError("ML-Ergebnisse passen nicht zu " + variant)
            if not candidates:
                # Leer ist kein vom Modell erkanntes BilderOhneSchilder.
                values = [frame_number, saved_name, elapsed, variant, 0, "",
                          "KEINE_AUSSCHNITTE"] + [""] * 8
                values += ml_result_values(None)
                file.write(";".join(str(value) for value in values) + "\n")
            for index, candidate in enumerate(candidates):
                rect, roi, color_index = candidate
                values = [frame_number, saved_name, elapsed, variant, index + 1,
                          COLOR_NAMES[color_index], "OK"]
                values += list(rect) + list(roi) + ml_result_values(results[index])
                file.write(";".join(str(value) for value in values) + "\n")


def draw_ml_labels(target, candidates, results):
    if len(candidates) != len(results):
        raise ValueError("ML-Beschriftungen passen nicht zu den Ausschnitten")
    for index, candidate in enumerate(candidates):
        rect, roi, color_index = candidate
        result = results[index]
        text = "%d:%s %.3f" % (index + 1, result["label"], result["score"])
        text_width = min(target.width(), len(text) * 8 + 4)
        text_x = min(roi[0], target.width() - text_width)
        text_y = roi[1] - 12 if roi[1] >= 26 else roi[1] + 16
        text_y = min(target.height() - 12, text_y)
        target.draw_rectangle((text_x, text_y, text_width, 12),
                              color=BLACK, fill=True)
        target.draw_string((text_x + 2, text_y + 1), text,
                           color=BOX_COLORS[color_index])


def make_preview(preview, frame, candidates_a, candidates_b, variant, read_ok,
                 results_a=None, results_b=None):
    if variant not in ("A", "B"):
        raise ValueError("Vorschauvariante muss A oder B sein")
    # trace("V1: vor Vollbildkopie fuer " + variant)
    step_us = time.ticks_us()
    copy_rgb565(preview, frame)
    # trace_done("V2: Vollbildkopie fertig; vor ROI-Rechtecken", step_us)
    selected = candidates_a if variant == "A" else candidates_b
    draw_candidates(preview, selected)
    results = results_a if variant == "A" else results_b
    if results is not None:
        draw_ml_labels(preview, selected, results)
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


def create_timing_logs(folder):
    header = ["kamerabild", "bildsatz", "laufzeit_ms", "tof_ok",
              "A_anzahl", "B_anzahl"]
    header += [key + "_ms" for key in FRAME_TIME_KEYS]
    header += ["tof_%d_ms" % n for n in range(1, AVERAGE_FRAMES + 1)]
    header += ["bild_gc_ms"]
    with open(folder + "/zeitmessung.csv", "w") as file:
        file.write(";".join(header) + "\n")
    header = ["kamerabild", "farbe", "close_radius", "A_anzahl", "B_anzahl"]
    header += [key + "_ms" for key in COLOR_TIME_KEYS]
    with open(folder + "/zeitmessung_farben.csv", "w") as file:
        file.write(";".join(header) + "\n")


def append_timings(folder, frame_number, saved_name, elapsed, read_ok,
                   candidates_a, candidates_b, timings):
    # Erst nach Messserie, Verarbeitung, Vorschau und bildweitem GC schreiben.
    # Die Protokollierung selbst ist nicht in den A/B-Zeiten enthalten.
    ensure_space()
    values = [frame_number, saved_name, elapsed, int(read_ok),
              len(candidates_a), len(candidates_b)]
    values += ["%.3f" % (timings.get(key, 0) / 1000.0)
               for key in FRAME_TIME_KEYS]
    values += ["%.3f" % (timings.get("tof_%d" % n, 0) / 1000.0)
               for n in range(1, AVERAGE_FRAMES + 1)]
    values += ["%.3f" % (timings.get("bild_gc", 0) / 1000.0)]
    with open(folder + "/zeitmessung.csv", "a") as file:
        file.write(";".join(str(value) for value in values) + "\n")
    with open(folder + "/zeitmessung_farben.csv", "a") as file:
        for color_index, row in enumerate(timings["farben"]):
            values = [frame_number, COLOR_NAMES[color_index],
                      CLOSE_RADIUS[color_index], row["A_anzahl"], row["B_anzahl"]]
            values += ["%.3f" % (row.get(key, 0) / 1000.0)
                       for key in COLOR_TIME_KEYS]
            file.write(";".join(str(value) for value in values) + "\n")


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
                    ("Version", "M13-GC ML"), ("Test", TEST_NAME),
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
                    ("SAVE_TIMINGS", SAVE_TIMINGS),
                    ("MODEL_PATH", MODEL_PATH),
                    ("MODEL_INPUT_SIZE", MODEL_INPUT_SIZE),
                    ("LABELS", LABELS),
                    ("SAVE_ML_RESULTS", SAVE_ML_RESULTS),
                    ("DISPLAY_MODE", DISPLAY_MODE),
                    ("SAVE_IMAGES", SAVE_IMAGES),
                    ("SAVE_TOF_PREVIEW", SAVE_TOF_PREVIEW),
                    ("SAVE_TOF_VALUES", SAVE_TOF_VALUES),
                    ("SAVE_INTERVAL_MS", SAVE_INTERVAL_MS),
                    ("MAX_SAVED_SETS", MAX_SAVED_SETS),
                    ("LOOP_PAUSE_MS", LOOP_PAUSE_MS),
                ):
                    file.write("%s: %s\n" % (key, value))
                file.write("A: LAB -> Schliessen -> Lochfuellung -> Suche\n")
                file.write("B: gleiche Farbmaske AND ToF -> Suche\n")
                file.write("Unbekannt/ToF-Lesefehler: sperren; ausserhalb ToF: freigeben\n")
                file.write("Keine ToF-Zielstatus- oder Signalstaerkedaten\n")
                file.write("ML-Inferenz fuer jeden Original-ROI von A und B; keine weitere Formpruefung\n")
                file.write("ML: Area-Resampling auf 48x48 RGB; Float-RGB 0..255, keine Normalisierung\n")
                file.write("Ein Modell fuer A/B; identische ROIs werden separat bewertet, kein Cache\n")
                file.write("Keine ML-Score-Schwelle; auch BilderOhneSchilder bleibt erhalten\n")
                file.write("Modellwerte sind rohe Ausgabewerte; keine manuell vorgegebenen Sollklassen\n")
                file.write("ml_ergebnisse.csv: jede ROI je Kamerabild, auch ohne Bildspeicherung\n")
                file.write("modellwert_00..12 entspricht der Reihenfolge von LABELS\n")
                file.write("KEINE_AUSSCHNITTE bezeichnet eine leere Kandidatenliste, keine Modellvorhersage\n")
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
                file.write("Kamerabild -> frame.flush -> ToF-Serie -> Masken -> ML A/B -> Anzeige/Speicherung\n")
                file.write("Keine Mittelungswerte aus vorherigen Kamerabildern\n")
                file.write("tof_zonen.csv: messwert_1_mm bis messwert_N_mm, aeltester bis neuester\n")
                file.write("Einzelwerte bleiben unveraendert; Mittelwert/Spannweite nur aus endlichen Werten >= %s mm\n" % MIN_VALID_DISTANCE_MM)
                file.write("Verbleibende gueltige Werte verwenden; ohne gueltigen Wert UNBEKANNT\n")
                file.write("Gueltige Nahanzahl >= Fernanzahl: NAH, freigeben\n")
                file.write("Fernanzahl > Nahanzahl: FERN, sofort sperren\n")
                file.write("Keine Spannweitenpruefung, keine Hysterese, keine Bestaetigung durch weitere Kamerabilder\n")
                file.write("Mittelwert/Spannweite nur Protokollwerte; Zonenentscheidung aus Nah-/Fernanzahl\n")
                file.write("Leere Einzelwertfelder: noch kein Wert oder ToF-Lesefehler\n")
                file.write("tof_einzelwerte.csv: jede Auslesung; kamerabild entspricht messwerte.csv\n")
                file.write("tof_nr: Auslesung 1 bis AVERAGE_FRAMES innerhalb dieses Kamerabildes\n")
                file.write("tof_laufzeit_ms: Zeit seit Schleifenstart, direkt nach ToF-Auslesung\n")
                file.write("tof_ok=1: 64 Werte gelesen, keine Aussage ueber Zielstatus/Signalstaerke\n")
                file.write("tof_ok=0: keine vollstaendige Auslesung, alle 64 Zonenfelder leer\n")
                file.write("messwerte.csv tof_ok: alle Auslesungen der Bildserie erfolgreich\n")
                file.write("ToF-Protokoll laeuft nach dem Bildlimit bis zum Stop oder Speicherlimit weiter\n")
                file.write("zeitmessung.csv: eine Zeile pro Kamerabild; leeres bildsatz-Feld ohne gespeicherte Aufnahme\n")
                file.write("zeitmessung_farben.csv: drei Zeilen pro Kamerabild; alle Zeiten in Millisekunden\n")
                file.write("A_nur_farbe: gemeinsamer Farbanteil + A-clear + A-Zweig + A-Sortierung\n")
                file.write("B_ohne_tof: gemeinsamer Farbanteil + B-clear + B-Zweig + B-Sortierung\n")
                file.write("B_mit_tof: B_ohne_tof + tof_lesen + tof_maske\n")
                file.write("tof_maske umfasst Mittelung, ZoneGate und Maskenzeichnung; nicht nochmals addieren\n")
                file.write("Gemeinsamer Farbanteil enthaelt binary, close, Hilfsbild, Lochfuellung und Kopie\n")
                file.write("A/B enthalten jetzt auch eigenes Clear, Union und Sortierung; gemeinsamer Anteil nicht doppelt addieren\n")
                file.write("AB_vergleich_wall: tatsaechliche gemeinsame Ausfuehrung beider Varianten ohne ToF\n")
                file.write("A_ml_gesamt/B_ml_gesamt: ROI-Kopie, Resize, Tensor, predict, Auswertung und Schleife, ohne manuelle GC\n")
                file.write("Manuelle GC einmal am Ende jedes Kamerabildes, nach A/B, Speicherung und Anzeige\n")
                file.write("bild_gc_ms: Dauer dieser gemeinsamen GC, ausserhalb der A/B-Verarbeitung\n")
                file.write("A_ml_gc/B_ml_gc und ml_gc je ROI: 0; automatische GC bleibt aktiv\n")
                file.write("GC und Zeitprotokoll nutzen LOOP_PAUSE_MS; danach nur verbleibende Restpause\n")
                file.write("A_ml_inferenz/B_ml_inferenz: nur model.predict; in ml_gesamt bereits enthalten\n")
                file.write("A_mit_ml = A_nur_farbe + A_ml_gesamt\n")
                file.write("B_mit_tof_ml = B_mit_tof + B_ml_gesamt\n")
                file.write("AB_ml_wall: tatsaechliche ML-Ausfuehrung A dann B, ohne Modellladezeit\n")
                file.write("ml_protokoll: Schreiben des ML-Laufprotokolls, ausserhalb der A/B-Verarbeitung\n")
                file.write("bildspeicherung: bestehende JPEG/PNG/CSV-Speicherung; 0 ohne Speicherversuch\n")
                file.write("ablauf_bis_vorschau: Kamera bis fertige Liveanzeige, inklusive ToF-Protokoll, ML und Bildspeicherung\n")
                file.write("ML-/Zeit-Laufprotokoll, spaetere Terminalausgabe, Schluss-GC und Schleifenpause sind darin nicht enthalten\n")
                file.write("Feine Einzelzeiten und Sammelzeiten ueberlappen; nicht alle Spalten addieren\n")
            with open(folder + "/messwerte.csv", "w") as file:
                file.write("satz;kamerabild;laufzeit_ms;tof_ok;A_anzahl;B_anzahl;"
                           "gesperrte_zonen;unbekannt_unruhig;tof_lesen_ms;"
                           "tof_maske_ms;A_verarbeitung_ms;B_verarbeitung_ms\n")
            if SAVE_TOF_VALUES:
                with open(folder + "/tof_einzelwerte.csv", "w") as file:
                    columns = ["kamerabild", "tof_nr", "tof_laufzeit_ms", "tof_ok"]
                    columns += ["zone_%02d_mm" % zone for zone in range(64)]
                    file.write(";".join(columns) + "\n")
            if SAVE_TIMINGS:
                create_timing_logs(folder)
            if SAVE_ML_RESULTS:
                create_ml_log(folder)
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


def read_tof_batch(averager, frame_number, started, log_folder, timings=None):
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
        sample_us = time.ticks_diff(time.ticks_us(), sample_started)
        tof_us += sample_us
        if timings is not None:
            timings["tof_%d" % reading_number] = sample_us
        elapsed = time.ticks_diff(time.ticks_ms(), started)
        readings.append((raw, sample_ok, elapsed))
    # Erst NACH den fuenf Auslesungen mitteln und protokollieren.
    # So verzoegert die CSV-Speicherung die Messserie nicht.
    averaging_started = time.ticks_us()
    for raw, sample_ok, elapsed in readings:
        means, spreads = averager.update(raw, sample_ok)
    averaging_us = time.ticks_diff(time.ticks_us(), averaging_started)
    if timings is not None:
        timings["tof_lesen"] = tof_us
        timings["tof_mittelung"] = averaging_us
    logging_started = time.ticks_us() if log_folder else None
    if log_folder:
        try:
            for index, (raw, sample_ok, elapsed) in enumerate(readings):
                append_tof_reading(log_folder, frame_number, index + 1,
                                   elapsed, raw, sample_ok)
        except (OSError, MemoryError) as error:
            print("ToF-Protokoll beendet; Livevergleich laeuft weiter:", error)
            log_folder = None
    if timings is not None:
        timings["tof_protokoll"] = (
            time.ticks_diff(time.ticks_us(), logging_started)
            if logging_started is not None else 0
        )
    # Bei Lesefehlern in der Serie vorsichtshalber freigeben.
    read_ok = all(averager.read_ok_history)
    return (readings[-1][0], means, spreads, read_ok, tof_us,
            averaging_us, log_folder)


def save_set(folder, number, frame_number, elapsed, frame, preview,
             tof_mask, union_a, union_b, candidates_a, candidates_b,
             raw, means, counts, spreads, states, gate, read_ok,
             grid_x, grid_y, timing, averager, results_a, results_b):
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
        make_preview(preview, frame, candidates_a, candidates_b, variant, read_ok,
                     results_a, results_b)
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
                   "roi_x;roi_y;roi_breite;roi_hoehe;datei;"
                   + ";".join(ml_result_header()) + "\n")
        for variant, candidates, results in (
            ("A", candidates_a, results_a), ("B", candidates_b, results_b)
        ):
            if len(candidates) != len(results):
                raise ValueError("ML-Ergebnisse passen nicht zu " + variant)
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
                values += ml_result_values(results[index])
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
    print("AE3 Farb-/ToF-Vergleich M13-GC | ML A/B | GC je Kamerabild")
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
    print("A/B verwenden dasselbe Kamerabild und dasselbe ML-Modell")
    print("Vollbildanzeige:", DISPLAY_MODE, "| gespeichert werden A und B")
    print("Pro Kamerabild:", AVERAGE_FRAMES,
          "ToF-Auslesungen direkt nacheinander; Mittelung nur fuer dieses Bild")
    print("Werte unter", MIN_VALID_DISTANCE_MM,
          "mm verwerfen; Nahanzahl >= Fernanzahl gibt frei; UNBEKANNT sperren")
    print("GC: einmal am Bildende | Pausenbudget:", LOOP_PAUSE_MS,
          "ms inklusive GC und Zeitprotokoll")
    if SAVE_IMAGES and SAVE_TOF_PREVIEW:
        print("Je Bildsatz: tof_raster.jpg mit Zonenwerten und R/C ab 0")

    model = load_model()
    averager, gate = ZoneAverage(AVERAGE_FRAMES), ZoneGate()
    tof_mask = image.Image(width, height, image.GRAYSCALE)
    union_a = image.Image(width, height, image.GRAYSCALE)
    union_b = image.Image(width, height, image.GRAYSCALE)
    filled = image.Image(width + 2, height + 2, image.GRAYSCALE)
    preview = image.Image(width, height, image.RGB565)
    folder = (prepare_save_dir(width, height, grid_x, grid_y)
              if SAVE_IMAGES or SAVE_TOF_VALUES or SAVE_TIMINGS
              or SAVE_ML_RESULTS else None)
    if folder:
        print("Speicherordner:", folder)
    log_folder = folder if SAVE_TOF_VALUES else None
    timing_folder = folder if SAVE_TIMINGS else None
    ml_folder = folder if SAVE_ML_RESULTS else None
    if log_folder:
        print("Alle ToF-Auslesungen:", log_folder + "/tof_einzelwerte.csv")
    if timing_folder:
        print("Zeitmessungen:", timing_folder + "/zeitmessung.csv")
        print("Zeiten je Farbe:", timing_folder + "/zeitmessung_farben.csv")
    if ml_folder:
        print("ML-Ergebnisse fuer A und B:", ml_folder + "/ml_ergebnisse.csv")
    if not SAVE_IMAGES:
        folder = None
    saved, frame_number = 0, 0
    started = time.ticks_ms()
    last_save = last_report = started
    print("4: Hauptschleife startet")
    while True:
        cycle_started = time.ticks_us()
        timings = {}
        saved_name = ""
        frame_number += 1
        TRACE_FRAME = frame_number
        # trace("L1: vor cam.snapshot")
        step_us = time.ticks_us()
        frame = cam.snapshot()
        record_duration(timings, "kamera", step_us)
        # trace_done("L2: cam.snapshot fertig; vor fruehem frame.flush", step_us)
        # Bewaehrte Reihenfolge fuer den AE3-Kamera-/ToF-Dauerbetrieb.
        step_us = time.ticks_us()
        frame.flush()
        record_duration(timings, "fruehes_flush", step_us)
        # trace("L3: fruehes frame.flush fertig; vor tof.read_depth")
        # Fuer das aktuelle Kamerabild fuenfmal hintereinander auslesen.
        # read_tof_batch setzt dabei alle Mittelungswerte zurueck.
        raw, means, spreads, read_ok, tof_us, averaging_us, log_folder = read_tof_batch(
            averager, frame_number, started, log_folder, timings
        )
        mask_started = time.ticks_us()
        # trace("L5: Mittelung fertig; vor ZoneGate")
        step_us = time.ticks_us()
        states = gate.update(averager, read_ok)
        record_duration(timings, "tof_zonegate", step_us)
        # trace("L6: ZoneGate fertig; vor ToF-Maske zeichnen")
        step_us = time.ticks_us()
        build_tof_mask(tof_mask, gate.blocked, grid_x, grid_y)
        record_duration(timings, "tof_maske_zeichnen", step_us)
        trace("L7: ToF-Maske fertig")
        mask_us = averaging_us + time.ticks_diff(time.ticks_us(), mask_started)
        timings["tof_maske"] = mask_us
        # trace("L8: vor Farb-/ToF-Maskenvergleich")
        candidates_a, candidates_b, a_us, b_us = compare_masks(
            frame, tof_mask, filled, union_a, union_b, timings
        )
        timing = (tof_us / 1000.0, mask_us / 1000.0,
                  a_us / 1000.0, (b_us + tof_us + mask_us) / 1000.0)
        timings["B_mit_tof"] = b_us + tof_us + mask_us
        trace("L9: Maskenvergleich fertig")
        # ML erst nach beiden Masken; das Kamerabild ist noch unmarkiert.
        # Modell- und ROI-Ergebnisse bleiben getrennt fuer A und B.
        ml_started = time.ticks_us()
        results_a = classify_candidates(model, frame, candidates_a, "A", timings)
        results_b = classify_candidates(model, frame, candidates_b, "B", timings)
        record_duration(timings, "AB_ml_wall", ml_started)
        timings["A_mit_ml"] = a_us + timings["A_ml_gesamt"]
        timings["B_mit_tof_ml"] = timings["B_mit_tof"] + timings["B_ml_gesamt"]
        now = time.ticks_ms()
        trace("L10: Speicherintervall pruefen")
        if folder and time.ticks_diff(now, last_save) >= SAVE_INTERVAL_MS:
            # trace("L11: vor Speicherung")
            saving_started = time.ticks_us()
            try:
                name = save_set(folder, saved + 1, frame_number,
                                time.ticks_diff(now, started), frame, preview,
                                tof_mask, union_a, union_b, candidates_a,
                                candidates_b, raw, means, averager.counts,
                                spreads, states, gate, read_ok, grid_x, grid_y,
                                timing, averager, results_a, results_b)
                saved_name = name
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
            record_duration(timings, "bildspeicherung", saving_started)
        # trace("L12: vor Vollbildvorschau")
        preview_started = time.ticks_us()
        make_preview(preview, frame, candidates_a, candidates_b,
                     DISPLAY_MODE, read_ok, results_a, results_b)
        # Erst jetzt darf das saubere Kamerabild durch die Anzeige ersetzt werden.
        # trace("L13: Anzeigemodus fertig; vor Vorschau in Kamerabild kopieren")
        copy_rgb565(frame, preview)
        # trace("L14: Kopieren fertig; vor abschliessendem frame.flush")
        frame.flush()
        trace("L15: abschliessendes frame.flush fertig")
        record_duration(timings, "live_vorschau", preview_started)
        record_duration(timings, "ablauf_bis_vorschau", cycle_started)
        if ml_folder:
            ml_log_started = time.ticks_us()
            try:
                append_ml_results(ml_folder, frame_number, saved_name,
                                  time.ticks_diff(time.ticks_ms(), started),
                                  candidates_a, candidates_b, results_a, results_b)
            except (OSError, MemoryError) as error:
                print("ML-Protokoll beendet; Livevergleich laeuft weiter:", error)
                ml_folder = None
            record_duration(timings, "ml_protokoll", ml_log_started)
        if frame_number == 1 or time.ticks_diff(now, last_report) >= 1000:
            print("Bild", frame_number, "| A:", len(candidates_a),
                  "| B:", len(candidates_b), "| gesperrte Zonen:",
                  sum(gate.blocked), "| ToF-Auslesungen:", AVERAGE_FRAMES,
                  "| ToF ms:", timing[0], "| A/B Verarbeitung ms:",
                  timing[2], timing[3])
            print("Schrittzeiten ms | binary: %.1f | close: %.1f | Loecher: %.1f | A-Suche: %.1f | B-Suche: %.1f" % (
                timings["binary"] / 1000.0, timings["close"] / 1000.0,
                timings["lochfuellung"] / 1000.0,
                timings["A_flachensuche"] / 1000.0,
                timings["B_flachensuche"] / 1000.0,
            ))
            print("ML ms | A: %.3f | B: %.3f | Gesamt mit Vorverarbeitung ms: %.3f %.3f" % (
                timings["A_ml_gesamt"] / 1000.0,
                timings["B_ml_gesamt"] / 1000.0,
                timings["A_mit_ml"] / 1000.0,
                timings["B_mit_tof_ml"] / 1000.0,
            ))
            for variant, results in (("A", results_a), ("B", results_b)):
                for index, result in enumerate(results):
                    print("ML %s_%02d | %s | Wert %.6f | predict %.3f ms" % (
                        variant, index + 1, result["label"], result["score"],
                        result["inferenz_us"] / 1000.0,
                    ))
            last_report = now
        # Ein manueller GC nach Verarbeitung, Speicherung und Anzeige dieses
        # Kamerabildes. Die GC-Zeit gehoert keiner einzelnen A/B-Variante.
        pause_started = time.ticks_ms()
        # trace("L16: vor gc.collect")
        step_us = time.ticks_us()
        gc.collect()
        record_duration(timings, "bild_gc", step_us)
        # Erst jetzt kann zeitmessung.csv die GC-Zeit dieses Bildes enthalten.
        if timing_folder:
            try:
                append_timings(timing_folder, frame_number, saved_name,
                               time.ticks_diff(time.ticks_ms(), started),
                               read_ok, candidates_a, candidates_b, timings)
            except (OSError, MemoryError) as error:
                print("Zeitprotokoll beendet; Livevergleich laeuft weiter:", error)
                timing_folder = None
        # trace("L17: Speicherbereinigung fertig; vor Schleifenpause")
        # GC und Zeitprotokoll nutzen die bereits vorgesehene Pause.
        # Bei ueberschrittenem Budget niemals eine negative Schlafzeit setzen.
        pause_remaining = max(0, LOOP_PAUSE_MS -
                              time.ticks_diff(time.ticks_ms(), pause_started))
        time.sleep_ms(pause_remaining)
        trace("L18: Pause beendet; Bilddurchlauf fertig")


if __name__ == "__main__":
    main()
