from pathlib import Path
from PIL import Image, UnidentifiedImageError


image_folder = Path(
    r"C:\Users\nicoh\OneDrive\Desktop\Matlabcode_Teamprojekt"
    r"\09_Trainingsdaten\AusgeschnittenFuerTraining"
    r"\99_Trainingsdatensaetze"
    r"\03_13Kat_Alle Schilder"
)

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}

two_channel_images = []
unreadable_images = []


if not image_folder.exists():
    raise FileNotFoundError(
        f"Der Bildordner wurde nicht gefunden:\n{image_folder}"
    )


print("Suche nach Bildern mit zwei Kanälen ...\n")


for image_path in image_folder.rglob("*"):

    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in valid_extensions:
        continue

    try:
        with Image.open(image_path) as image:
            image.load()

            mode = image.mode
            bands = image.getbands()
            number_of_channels = len(bands)

            if number_of_channels == 2:
                two_channel_images.append(image_path)

                print("Zweikanalbild gefunden:")
                print(f"  Datei:   {image_path}")
                print(f"  Modus:   {mode}")
                print(f"  Kanäle:  {bands}")
                print(f"  Größe:   {image.size}")
                print()

    except UnidentifiedImageError:
        unreadable_images.append(image_path)
        print(f"Nicht als Bild lesbar: {image_path}")

    except Exception as error:
        unreadable_images.append(image_path)
        print(f"Fehler beim Lesen von {image_path}:")
        print(f"  {error}\n")


print("----------------------------------------")
print("Auswertung abgeschlossen")
print("----------------------------------------")
print(f"Zweikanalbilder gefunden: {len(two_channel_images)}")
print(f"Nicht lesbare Bilder:     {len(unreadable_images)}")


if two_channel_images:
    print("\nBetroffene Dateien:")

    for image_path in two_channel_images:
        print(image_path)
else:
    print("\nEs wurden keine Zweikanalbilder gefunden.")