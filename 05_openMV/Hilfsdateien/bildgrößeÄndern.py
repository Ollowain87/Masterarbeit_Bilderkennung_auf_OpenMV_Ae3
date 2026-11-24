import sensor, image, time

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.HD)            # ≥200×200
sensor.skip_frames(time=2000)

src = sensor.snapshot()                      # z.B. 200×200
src.save("original.bmp")
# Puffer für 48×48 RGB565
dst = image.Image(48, 48, sensor.RGB565)

ow, oh = src.width(), src.height()
for y in range(48):
    for x in range(48):
        sx = x * ow // 48
        sy = y * oh // 48
        dst.set_pixel(x, y, src.get_pixel(sx, sy))

dst.save("scaled.bmp")
