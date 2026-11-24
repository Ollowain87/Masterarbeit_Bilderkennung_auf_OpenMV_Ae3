import sensor, image, time

def capture_and_scale(save_original=True, save_scaled=True, scaled_size=(48, 48)):
    # Kamera initialisieren
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.HD)
    sensor.skip_frames(time=2000)

    # Bild aufnehmen
    src = sensor.snapshot()
    if save_original:
        src.save("original.bmp")

    # Zielbild erzeugen
    dst = image.Image(scaled_size[0], scaled_size[1], sensor.RGB565)

    ow, oh = src.width(), src.height()
    for y in range(scaled_size[1]):
        for x in range(scaled_size[0]):
            sx = x * ow // scaled_size[0]
            sy = y * oh // scaled_size[1]
            dst.set_pixel(x, y, src.get_pixel(sx, sy))

    if save_scaled:
        dst.save("scaled.bmp")

    return dst  # Gibt das skalierte Bild zurück
