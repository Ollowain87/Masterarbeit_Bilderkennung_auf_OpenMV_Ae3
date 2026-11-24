# OpenMV AE3: Praxistest P1 - ToF-Zonen mit gemittelten Millimeterwerten.
# In der OpenMV IDE oeffnen und starten.
# Bewaehrte Reihenfolge: Kamerabild -> flush -> ToF -> Zeichnen -> flush.
# Das Raster ist eine vorlaeufige Zuordnung, keine pixelgenaue Kalibrierung.

import csi
import os
import time
import tof

# AE3 mit PAG7936: csi.VGA liefert hier 640 x 400 Pixel.
GRID_W = 428
GRID_H = 404
GRID_SHIFT_X = 10
GRID_SHIFT_Y = -11

# Optional vor einem Lauf eintragen. None = Referenzabstand unbekannt.
# Beispiel: TEST_NAME = "mitte_50cm" und REFERENCE_MM = 500
# Diese Angaben beschriften die CSV; sie korrigieren keine Messwerte.
TEST_NAME = "freier_test"
REFERENCE_MM = None

# Gleitender Mittelwert pro Zone ueber die letzten N gelesenen Messungen.
# 1 = keine Mittelung; groessere Werte glaetten staerker, reagieren langsamer.
# Am Anfang werden die bereits vorhandenen Werte verwendet.
# Werte <= 0, NaN und Unendlich werden nicht mitgemittelt.
AVERAGE_FRAMES = 5

# Anzeige und rote Rahmen verwenden dieselben gemittelten Abstaende.
# Der Rahmen braucht weiterhin zwei Bestaetigungen zum Ein-/Ausschalten.
NEAR_MAX_MM = 500
NEAR_ON_FRAMES = 2
NEAR_OFF_FRAMES = 2

SAVE_IMAGES = True
SAVE_ROOT = "/flash/tof_praxistest"
SAVE_INTERVAL_MS = 2000
MAX_SAVED_IMAGES = 100
JPEG_QUALITY = 60
MIN_FREE_BYTES = 1000000
LOOP_PAUSE_MS = 200
DEBUG_TRACE = False

YELLOW = (255, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)
BLACK = (0, 0, 0)


def trace(number, message):
    if DEBUG_TRACE:
        print("[Bild %d] %s" % (number, message))


def usable_number(distance):
    # Nur numerische Plausibilitaet, kein Ersatz fuer den ToF-Zielstatus.
    return distance > 0 and distance < float("inf")


def distance_text(distance):
    if not usable_number(distance):
        return "--"
    mm = int(distance + 0.5)
    return str(mm) if mm <= 9999 else ">9999"


class ZoneAverage:
    """Zeitlicher Mittelwert je Zone; benachbarte Zonen bleiben getrennt."""

    def __init__(self, frames):
        if not isinstance(frames, int) or frames < 1:
            raise ValueError("AVERAGE_FRAMES muss eine positive ganze Zahl sein")
        self.frames = frames
        self.history = [[0.0] * 64 for _ in range(frames)]
        self.sums = [0.0] * 64
        self.counts = [0] * 64
        self.next_slot = 0

    def update(self, depth):
        # Jeder Platz gehoert zu einer Messung; auch ungueltige Werte altern.
        slot = self.history[self.next_slot]
        averaged = [0.0] * 64
        for zone in range(64):
            if slot[zone] > 0:
                self.sums[zone] -= slot[zone]
                self.counts[zone] -= 1

            value = depth[zone]
            slot[zone] = value if usable_number(value) else 0.0
            if slot[zone] > 0:
                self.sums[zone] += slot[zone]
                self.counts[zone] += 1

            if self.counts[zone]:
                averaged[zone] = self.sums[zone] / self.counts[zone]
            else:
                # Kein nutzbarer Wert mehr im Fenster: Anzeige "--".
                self.sums[zone] = 0.0

        self.next_slot = (self.next_slot + 1) % self.frames
        return averaged


def update_near_state(depth, active, on_counts, off_counts):
    near_zones = []
    for zone in range(64):
        distance = depth[zone]
        near = usable_number(distance) and distance <= NEAR_MAX_MM
        if near:
            near_zones.append(zone)
            on_counts[zone] = min(NEAR_ON_FRAMES, on_counts[zone] + 1)
            off_counts[zone] = 0
            if on_counts[zone] >= NEAR_ON_FRAMES:
                active[zone] = True
        else:
            off_counts[zone] = min(NEAR_OFF_FRAMES, off_counts[zone] + 1)
            on_counts[zone] = 0
            if off_counts[zone] >= NEAR_OFF_FRAMES:
                active[zone] = False
    return near_zones


def draw_grid_and_values(img, depth, active):
    width = img.width()
    height = img.height()
    left = max(0, GRID_X)
    right = min(width, GRID_X + GRID_W)
    top = max(0, GRID_Y)
    bottom = min(height, GRID_Y + GRID_H)

    if left >= right or top >= bottom:
        return

    for n in range(9):
        x = GRID_X + n * GRID_W // 8
        y = GRID_Y + n * GRID_H // 8
        if 0 <= x < width:
            img.draw_line((x, top, x, bottom - 1), color=YELLOW)
        if 0 <= y < height:
            img.draw_line((left, y, right - 1, y), color=YELLOW)

    for zone in range(64):
        row = zone // 8
        col = zone % 8
        x0 = max(0, GRID_X + col * GRID_W // 8)
        x1 = min(width, GRID_X + (col + 1) * GRID_W // 8)
        y0 = max(0, GRID_Y + row * GRID_H // 8)
        y1 = min(height, GRID_Y + (row + 1) * GRID_H // 8)
        if x1 - x0 < 44 or y1 - y0 < 27:
            continue

        if active[zone]:
            img.draw_rectangle((x0, y0, x1 - x0, y1 - y0),
                               color=RED, thickness=2)

        # Kleine schwarze Textfelder fuer lesbare Werte; restliches Bild frei.
        text = distance_text(depth[zone])
        img.draw_rectangle((x0 + 3, y0 + 3, 24, 10),
                           color=BLACK, fill=True)
        img.draw_string((x0 + 3, y0 + 3), "Z%02d" % zone, color=GRAY)
        img.draw_rectangle((x0 + 3, y0 + 15, len(text) * 8, 10),
                           color=BLACK, fill=True)
        img.draw_string((x0 + 3, y0 + 15), text,
                        color=WHITE if usable_number(depth[zone]) else GRAY)

    # Legende im freien Streifen links neben dem Raster.
    if left >= 92:
        img.draw_rectangle((2, 2, 90, 80), color=BLACK, fill=True)
        img.draw_string((5, 5), "ToF P1", color=WHITE)
        img.draw_string((5, 18), "Werte: mm", color=WHITE)
        img.draw_string((5, 31), "Rot <=%d" % NEAR_MAX_MM, color=RED)
        img.draw_string((5, 44), "--: fehlt", color=GRAY)
        img.draw_string((5, 57), "Mittel:%d" % AVERAGE_FRAMES, color=WHITE)
        img.draw_string((5, 70), "Z00-Z63", color=GRAY)


def prepare_save_dir(width, height):
    try:
        if "tof_praxistest" not in os.listdir("/flash"):
            os.mkdir(SAVE_ROOT)
        existing = os.listdir(SAVE_ROOT)
        for run in range(1, 1000):
            name = "lauf_%03d" % run
            if name not in existing:
                folder = SAVE_ROOT + "/" + name
                os.mkdir(folder)
                header = "bild;laufzeit_ms;roh_nah;stabil_rot"
                for zone in range(64):
                    header += ";zone_%02d_mm" % zone
                header += ";mittel_nah"
                for zone in range(64):
                    header += ";zone_%02d_mittel_mm" % zone
                with open(folder + "/messwerte.csv", "w") as file:
                    file.write(header + ";test;referenz_mm\n")
                with open(folder + "/parameter.txt", "w") as file:
                    file.write("Test: %s\nReferenz_mm: %s\n" %
                               (TEST_NAME, REFERENCE_MM))
                    file.write("Bild: %d x %d\n" % (width, height))
                    file.write("GRID_W/H: %d / %d\n" % (GRID_W, GRID_H))
                    file.write("GRID_SHIFT_X/Y: %d / %d\n" %
                               (GRID_SHIFT_X, GRID_SHIFT_Y))
                    file.write("GRID_X/Y: %d / %d\n" % (GRID_X, GRID_Y))
                    file.write("ToF: vflip=True, hmirror=True\n")
                    file.write("Gleitender Mittelwert: letzte %d Messungen je Zone\n" %
                               AVERAGE_FRAMES)
                    file.write("Mittelung: nur positive endliche Werte; leer = --\n")
                    file.write("Anzeige und Rot: Mittelwerte; CSV: Roh- und Mittelwerte\n")
                    file.write("Rot: 0 < Mittelwert <= %d mm, ein/aus: %d/%d\n" %
                               (NEAR_MAX_MM, NEAR_ON_FRAMES, NEAR_OFF_FRAMES))
                return folder
        print("Bildspeicherung: Alle Laufnummern 001-999 belegt")
    except OSError as err:
        print("Bildspeicherung nicht verfuegbar:", err)
    return None


def free_flash_bytes():
    status = os.statvfs("/flash")
    return status[0] * status[3]


def save_frame(img, depth, mean_depth, raw_near, mean_near, active,
               folder, number, elapsed_ms):
    basename = "bild_%03d.jpg" % number
    img.save(folder + "/" + basename, quality=JPEG_QUALITY)
    stable = [str(zone) for zone in range(64) if active[zone]]
    test_name = TEST_NAME.replace(";", "_").replace("\n", " ").replace("\r", " ")
    reference = "" if REFERENCE_MM is None else str(REFERENCE_MM)
    row = (basename + ";" + str(elapsed_ms) + ";" +
           " ".join(str(zone) for zone in raw_near) + ";" +
           " ".join(stable) + ";" +
           ";".join(str(value) for value in depth) + ";" +
           " ".join(str(zone) for zone in mean_near) + ";" +
           ";".join(str(value) for value in mean_depth) + ";" +
           test_name + ";" + reference + "\n")
    with open(folder + "/messwerte.csv", "a") as file:
        file.write(row)
    return basename


print("AE3 ToF-Praxistest P1 - gemittelte Millimeterwerte")
averager = ZoneAverage(AVERAGE_FRAMES)
cam = csi.CSI()
cam.reset()
cam.pixformat(csi.RGB565)
cam.framesize(csi.VGA)
time.sleep_ms(1000)

print("Vor erstem Kamerabild")
img = cam.snapshot()
img.flush()
print("Tatsaechliche Bildgroesse:", img.width(), "x", img.height())
GRID_X = (img.width() - GRID_W) // 2 + GRID_SHIFT_X
GRID_Y = (img.height() - GRID_H) // 2 + GRID_SHIFT_Y
print("Kamera funktioniert; vor tof.init")
tof.init()
print("ToF initialisiert")
time.sleep_ms(1000)
print("Raster:", GRID_X, GRID_Y, GRID_W, GRID_H)
print("Gleitender Mittelwert je Zone: letzte", AVERAGE_FRAMES, "Messungen")
print("Anzeige und rote Rahmen verwenden Mittelwerte in mm")
print("Test:", TEST_NAME, "| Referenzabstand:", REFERENCE_MM)

save_folder = prepare_save_dir(img.width(), img.height()) if SAVE_IMAGES else None
if save_folder:
    print("Bilder und Messwerte:", save_folder)
else:
    print("Livebild ohne Speicherung")

active = [False] * 64
on_counts = [0] * 64
off_counts = [0] * 64
frame_number = 0
saved_images = 0
start_ms = time.ticks_ms()
last_report = start_ms
last_saved = start_ms

while True:
    frame_number += 1
    trace(frame_number, "A: vor cam.snapshot")
    img = cam.snapshot()
    trace(frame_number, "B: vor erster Bildausgabe")
    img.flush()

    trace(frame_number, "C: vor tof.read_depth")
    try:
        depth, dmin, dmax = tof.read_depth(
            vflip=True, hmirror=True, timeout=250
        )
    except RuntimeError as err:
        print("[Bild %d] ToF-Lesefehler:" % frame_number, err)
        time.sleep_ms(LOOP_PAUSE_MS)
        continue

    if len(depth) != 64:
        print("[Bild %d] Erwartet: 64 ToF-Werte, bekommen:" % frame_number,
              len(depth))
        time.sleep_ms(LOOP_PAUSE_MS)
        continue

    raw_near = [zone for zone in range(64)
                if usable_number(depth[zone]) and depth[zone] <= NEAR_MAX_MM]
    mean_depth = averager.update(depth)
    mean_near = update_near_state(mean_depth, active, on_counts, off_counts)
    trace(frame_number, "D: vor Raster und Millimeterwerten")
    draw_grid_and_values(img, mean_depth, active)
    trace(frame_number, "E: vor zweiter Bildausgabe")
    img.flush()
    trace(frame_number, "F: Bild mit Millimeterwerten ausgegeben")

    now = time.ticks_ms()
    if time.ticks_diff(now, last_report) >= 1000:
        values = [int(value + 0.5) for value in mean_depth if usable_number(value)]
        if values:
            print("ToF-Mittel:", min(values), "bis", max(values),
                  "mm | Zonen mit Mittelwert:", len(values), "/64")
        else:
            print("ToF-Mittel: keine positiven endlichen Werte im Fenster")
        last_report = now

    if save_folder and saved_images < MAX_SAVED_IMAGES:
        if time.ticks_diff(now, last_saved) >= SAVE_INTERVAL_MS:
            try:
                if free_flash_bytes() < MIN_FREE_BYTES:
                    print("Bildspeicherung gestoppt: weniger als 1 MB frei")
                    save_folder = None
                else:
                    name = save_frame(
                        img, depth, mean_depth, raw_near, mean_near, active, save_folder,
                        saved_images + 1, time.ticks_diff(now, start_ms)
                    )
                    saved_images += 1
                    last_saved = time.ticks_ms()
                    print("Gespeichert:", save_folder + "/" + name,
                          "(%d/%d)" % (saved_images, MAX_SAVED_IMAGES))
                    if saved_images == MAX_SAVED_IMAGES:
                        print("Aufnahmeserie beendet; Livebild laeuft weiter")
            except (OSError, MemoryError) as err:
                print("Bildspeicherung gestoppt:", err)
                save_folder = None

    time.sleep_ms(LOOP_PAUSE_MS)
