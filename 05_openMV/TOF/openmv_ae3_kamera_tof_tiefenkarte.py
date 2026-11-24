# OpenMV AE3: Kamerabild und separate 8x8-ToF-Tiefenkarte.
# Test D8 (VGA): Aufnehmen -> flush -> ToF lesen -> Farbfelder zeichnen -> flush.
# Zuerst in der IDE ausfuehren; noch nicht als main.py speichern.
# Die kleine Karte oben rechts ist noch nicht auf die Kamera kalibriert.

import csi
import time
import tof

CELL_SIZE = 32
MAP_SIZE = 8 * CELL_SIZE  # 256 x 256 Bildpixel; ToF bleibt bei 8 x 8 Zonen.
DISPLAY_MAX_MM = 2000
MIN_VALID_MM = 20
MAX_VALID_MM = 4000
DEBUG_TRACE = True

# Acht deutlich unterscheidbare Stufen fuer 0 bis unter 2000 mm.
# Bei DISPLAY_MAX_MM = 2000 umfasst jede Stufe 250 mm.
DEPTH_COLORS = (
    (255, 0, 0),      # Rot
    (255, 128, 0),    # Orange
    (255, 255, 0),    # Gelb
    (128, 255, 0),    # Hellgruen
    (0, 220, 0),      # Gruen
    (0, 255, 255),    # Tuerkis
    (0, 100, 255),    # Blau
    (180, 0, 255),    # Violett
)


def trace(number, step):
    if DEBUG_TRACE:
        print("[Bild %d] %s" % (number, step))


def valid(distance):
    return MIN_VALID_MM <= distance <= MAX_VALID_MM


def depth_color(distance):
    if not valid(distance):
        return (0, 0, 0)  # Ungueltige Zone schwarz.

    if distance >= DISPLAY_MAX_MM:
        return (255, 255, 255)  # Ab 2 m weiss (bis MAX_VALID_MM).

    band = int(distance) * len(DEPTH_COLORS) // DISPLAY_MAX_MM
    return DEPTH_COLORS[band]


def draw_map(img, depth, map_x, map_y):
    for zone in range(64):
        x = map_x + (zone % 8) * CELL_SIZE
        y = map_y + (zone // 8) * CELL_SIZE
        img.draw_rectangle(
            (x, y, CELL_SIZE, CELL_SIZE),
            color=depth_color(depth[zone]), fill=True
        )

    img.draw_rectangle(
        (map_x - 1, map_y - 1, MAP_SIZE + 2, MAP_SIZE + 2),
        color=(255, 255, 255)
    )
    legend_y = map_y + MAP_SIZE + 7
    img.draw_rectangle(
        (map_x - 2, legend_y - 3, MAP_SIZE + 4, 58),
        color=(0, 0, 0), fill=True
    )
    for band, color in enumerate(DEPTH_COLORS):
        img.draw_rectangle(
            (map_x + band * CELL_SIZE, legend_y, CELL_SIZE, 12),
            color=color, fill=True
        )
    img.draw_string((map_x, legend_y + 15),
                    "0 mm", color=(255, 255, 255))
    img.draw_string((map_x + MAP_SIZE - 28, legend_y + 15),
                    "2 m", color=(255, 255, 255))
    img.draw_string((map_x, legend_y + 28),
                    ">=2 m: weiss", color=(255, 255, 255))
    img.draw_string((map_x, legend_y + 41),
                    "X: ungueltig", color=(255, 255, 255))


print("AE3 Tiefenkarte: Diagnose D8 - VGA, acht Farbstufen")
cam = csi.CSI()
cam.reset()
cam.pixformat(csi.RGB565)
cam.framesize(csi.VGA)  # AE3 mit PAG7936: 640 x 400
print("Kamera konfiguriert")
time.sleep_ms(1000)

print("Vor erstem Kamerabild")
img = cam.snapshot()
img.flush()
print("Tatsaechliche Bildgroesse:", img.width(), "x", img.height())
print("Erstes Kamerabild funktioniert")

print("Vor tof.init")
tof.init()
print("tof.init abgeschlossen")
time.sleep_ms(1000)

map_x = img.width() - MAP_SIZE - 4
map_y = 40
frame_number = 0
last_report = time.ticks_ms()
print("Dauerbetrieb gestartet")

while True:
    frame_number += 1
    trace(frame_number, "F1: vor cam.snapshot")
    img = cam.snapshot()
    trace(frame_number, "F2: vor erster Bildausgabe")
    img.flush()
    trace(frame_number, "F3: vor tof.read_depth")

    try:
        depth, dmin, dmax = tof.read_depth(
            vflip=True, hmirror=True, timeout=250
        )
    except RuntimeError as err:
        print("[Bild %d] ToF-Lesefehler:" % frame_number, err)
        time.sleep_ms(200)
        continue

    if len(depth) != 64:
        print("[Bild %d] Unerwartete Zahl an ToF-Zonen:" % frame_number,
              len(depth))
        time.sleep_ms(200)
        continue

    trace(frame_number, "F4: ToF gelesen; vor Zeichnen der Karte")
    draw_map(img, depth, map_x, map_y)
    trace(frame_number, "F5: Karte fertig; vor zweiter Bildausgabe")
    img.flush()
    trace(frame_number, "F6: Bild mit Karte ausgegeben")

    now = time.ticks_ms()
    if time.ticks_diff(now, last_report) >= 1000:
        valid_values = [int(d) for d in depth if valid(d)]
        if valid_values:
            print("ToF:", min(valid_values), "bis", max(valid_values),
                  "mm | gueltige Zonen:", len(valid_values), "/64")
        else:
            print("ToF: keine gueltige Zone")
        last_report = now

    time.sleep_ms(200)
