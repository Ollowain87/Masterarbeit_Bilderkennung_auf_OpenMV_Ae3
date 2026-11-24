# OpenMV AE3: ToF-Zonen im Kamerabild sichtbar machen.
# Test K4: Bilder und Rohwerte speichern; rote Zonen zeitlich bestaetigen.
# In der IDE ausfuehren, noch nicht als main.py speichern.
# Kalibrierungstest: Ein helles A4-Blatt ca. 0.7-1.0 m entfernt halten,
# dahinter moeglichst wenig im Bereich 0.2-1.2 m.
# Gelbe Linien: GESCHAETZTE Grenzen des ToF-Rasters im Kamerabild.
# Rote Rahmen mit Index: ToF-Zonen mit einer nahen Messung.
# Rastergroesse und Versatz sind STARTWERTE, keine vermessene Kalibrierung.

import csi
import os
import time
import tof

# Erste Schaetzung aus 45 x 45 Grad ToF und dem AE3-Standardobjektiv.
# Auf der AE3 liefert csi.VGA tatsaechlich 640 x 400 Pixel.
# Das ToF-Sichtfeld kann etwas ueber die Bildhoehe hinausragen.
# Zum Abgleichen GRID_W/H sowie GRID_SHIFT_X/Y anpassen.
GRID_W = 428
GRID_H = 404
GRID_SHIFT_X = 10
GRID_SHIFT_Y = -11

# Fuer das A4-Blatt in etwa 0.7-0.9 m Entfernung.
NEAR_MIN_MM = 2
NEAR_MAX_MM = 500
NEAR_ON_FRAMES = 2   # Erst nach 3 passenden Messungen rot.
NEAR_OFF_FRAMES = 2  # Erst nach 3 anderen Messungen wieder aus.

SAVE_ROOT = "/flash/tof_abgleich"
SAVE_INTERVAL_MS = 2000
MAX_SAVED_IMAGES = 100
JPEG_QUALITY = 60
MIN_FREE_BYTES = 1000000
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
DEBUG_TRACE = False  # Bei Haengern fuer die Schritt-Ausgabe auf True setzen.


def trace(number, message):
    if DEBUG_TRACE:
        print("[Bild %d] %s" % (number, message))


def update_near_state(depth, active, on_counts, off_counts, last_near):
    raw_near = []
    for zone in range(64):
        distance = depth[zone]
        if NEAR_MIN_MM <= distance <= NEAR_MAX_MM:
            raw_near.append(zone)
            last_near[zone] = int(distance)
            on_counts[zone] = min(NEAR_ON_FRAMES, on_counts[zone] + 1)
            off_counts[zone] = 0
            if on_counts[zone] >= NEAR_ON_FRAMES:
                active[zone] = True
        else:
            off_counts[zone] = min(NEAR_OFF_FRAMES, off_counts[zone] + 1)
            on_counts[zone] = 0
            if off_counts[zone] >= NEAR_OFF_FRAMES:
                active[zone] = False
    return raw_near


def prepare_save_dir():
    # Jeder Start bekommt einen neuen Ordner; alte Bilder bleiben erhalten.
    try:
        if "tof_abgleich" not in os.listdir("/flash"):
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
                with open(folder + "/messwerte.csv", "w") as file:
                    file.write(header + "\n")
                return folder
    except OSError as err:
        print("Bildspeicherung nicht verfuegbar:", err)
    return None


def free_flash_bytes():
    status = os.statvfs("/flash")
    return status[0] * status[3]


def save_frame(img, depth, raw_near, stable_hits, folder,
               image_number, elapsed_ms):
    basename = "bild_%03d.jpg" % image_number
    img.save(folder + "/" + basename, quality=JPEG_QUALITY)
    stable_zones = [str(zone) for zone, mm in stable_hits]
    row = (basename + ";" + str(elapsed_ms) + ";" +
           " ".join(str(zone) for zone in raw_near) + ";" +
           " ".join(stable_zones) + ";" +
           ";".join(str(value) for value in depth) + "\n")
    with open(folder + "/messwerte.csv", "a") as file:
        file.write(row)
    return basename


def draw_grid_and_hits(img, active, last_near):
    width = img.width()
    height = img.height()
    left = max(0, GRID_X)
    right = min(width, GRID_X + GRID_W)
    top = max(0, GRID_Y)
    bottom = min(height, GRID_Y + GRID_H)

    # Nur Linien innerhalb des sichtbaren Kamerabilds zeichnen.
    if left < right and top < bottom:
        for n in range(9):
            x = GRID_X + n * GRID_W // 8
            y = GRID_Y + n * GRID_H // 8
            if 0 <= x < width:
                img.draw_line((x, top, x, bottom - 1), color=YELLOW)
            if 0 <= y < height:
                img.draw_line((left, y, right - 1, y), color=YELLOW)

    hits = []
    for index in range(64):
        if not active[index]:
            continue
        row = index // 8
        col = index % 8
        hits.append((index, last_near[index]))

        x0 = max(0, GRID_X + col * GRID_W // 8)
        x1 = min(width, GRID_X + (col + 1) * GRID_W // 8)
        y0 = max(0, GRID_Y + row * GRID_H // 8)
        y1 = min(height, GRID_Y + (row + 1) * GRID_H // 8)
        if x0 < x1 and y0 < y1:
            img.draw_rectangle((x0, y0, x1 - x0, y1 - y0),
                               color=RED, thickness=2)
            if x1 - x0 >= 20 and y1 - y0 >= 18:
                img.draw_string((x0 + 3, y0 + 3), str(index), color=RED)
    return hits


print("AE3 Kamera-ToF-Abgleich K4 - Bilder und stabile rote Zonen")
cam = csi.CSI()
cam.reset()
cam.pixformat(csi.RGB565)
cam.framesize(csi.VGA)  # AE3 mit PAG7936: 640 x 400
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
print("Raster-Startwerte:", GRID_X, GRID_Y, GRID_W, GRID_H)
print("Rote Zonen: %d bis %d mm, ein/aus: %d/%d Messungen" %
      (NEAR_MIN_MM, NEAR_MAX_MM, NEAR_ON_FRAMES, NEAR_OFF_FRAMES))

save_folder = prepare_save_dir()
if save_folder:
    print("Bilder und Messwerte:", save_folder)
else:
    print("Livebild laeuft ohne Speicherung weiter")

frame_number = 0
active = [False] * 64
on_counts = [0] * 64
off_counts = [0] * 64
last_near = [0] * 64
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
        time.sleep_ms(200)
        continue

    if len(depth) != 64:
        print("[Bild %d] Erwartet: 64 ToF-Werte, bekommen:" % frame_number,
              len(depth))
        time.sleep_ms(200)
        continue

    raw_near = update_near_state(depth, active, on_counts,
                                 off_counts, last_near)
    trace(frame_number, "D: vor Raster und Markierungen")
    hits = draw_grid_and_hits(img, active, last_near)
    trace(frame_number, "E: vor zweiter Bildausgabe")
    img.flush()
    trace(frame_number, "F: Bild mit Raster ausgegeben")

    now = time.ticks_ms()
    if time.ticks_diff(now, last_report) >= 1000:
        print("Roh nah:", raw_near,
              "| stabil rot (Index, letzter Nahwert mm):", hits)
        last_report = now

    if save_folder and saved_images < MAX_SAVED_IMAGES:
        if time.ticks_diff(now, last_saved) >= SAVE_INTERVAL_MS:
            try:
                if free_flash_bytes() < MIN_FREE_BYTES:
                    print("Bildspeicherung gestoppt: weniger als 1 MB frei")
                    save_folder = None
                else:
                    name = save_frame(
                        img, depth, raw_near, hits, save_folder,
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
    time.sleep_ms(200)
