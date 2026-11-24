from pathlib import Path
from PIL import Image
import shutil
import tensorflow as tf


image_folder = Path(
    r"C:\Users\nicoh\OneDrive\Desktop\Matlabcode_Teamprojekt"
    r"\09_Trainingsdaten\AusgeschnittenFuerTraining"
    r"\99_Trainingsdatensaetze"
    r"\03_13Kat_Alle Schilder"
)

# Backup liegt neben dem eigentlichen Datensatz
backup_folder = image_folder.parent / (
    image_folder.name + "_BMP_Backup"
)

converted_images = []
failed_images = []
tested_images = 0


def tensorflow_can_decode(image_path):
    """Prüft das Bild mit dem TensorFlow-BMP-Decoder."""

    image_data = tf.io.read_file(str(image_path))

    image = tf.io.decode_bmp(
        image_data,
        channels=3
    )

    # Erzwingt die tatsächliche Ausführung
    image.numpy()


print("Prüfe und konvertiere problematische BMP-Dateien ...\n")


for image_path in image_folder.rglob("*.bmp"):

    tested_images += 1

    try:
        # Bereits kompatible Bilder bleiben unverändert
        tensorflow_can_decode(image_path)
        continue

    except Exception:
        print(f"Problematisches BMP: {image_path}")

    try:
        # Relativen Pfad für das Backup beibehalten
        relative_path = image_path.relative_to(image_folder)
        backup_path = backup_folder / relative_path

        backup_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Original sichern
        shutil.copy2(
            image_path,
            backup_path
        )

        # Temporäre 24-Bit-RGB-BMP-Datei
        temporary_path = image_path.with_name(
            image_path.stem + "_temp.bmp"
        )

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")

            rgb_image.save(
                temporary_path,
                format="BMP"
            )

        # Prüfen, ob TensorFlow die neue Datei lesen kann
        tensorflow_can_decode(temporary_path)

        # Erst nach erfolgreicher Prüfung Original ersetzen
        temporary_path.replace(image_path)

        converted_images.append(image_path)

        print("  Erfolgreich als 24-Bit-RGB gespeichert.")

    except Exception as error:
        failed_images.append(image_path)

        print(f"  Konvertierung fehlgeschlagen: {error}")

        temporary_path = image_path.with_name(
            image_path.stem + "_temp.bmp"
        )

        if temporary_path.exists():
            temporary_path.unlink()


print("\n----------------------------------------")
print("Konvertierung abgeschlossen")
print("----------------------------------------")
print(f"Geprüfte BMP-Dateien:    {tested_images}")
print(f"Konvertierte Dateien:    {len(converted_images)}")
print(f"Fehlgeschlagene Dateien: {len(failed_images)}")
print(f"Backup-Ordner:           {backup_folder}")


if failed_images:
    print("\nNicht konvertierte Dateien:")

    for image_path in failed_images:
        print(image_path)