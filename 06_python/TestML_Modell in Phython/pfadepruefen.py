from pathlib import Path

image_folder = Path(
    r"C:\Users\nicoh\OneDrive\Desktop\Matlabcode_Teamprojekt"
    r"\09_Trainingsdaten\AusgeschnittenFuerTraining"
    r"\99_Trainingsdatensaetze"
    r"\03_13Kat_Alle Schilder"
)

if not image_folder.exists():
    raise FileNotFoundError(
        f"Der Ordner wurde nicht gefunden:\n{image_folder}"
    )

print("Pfade mit Umlauten oder Sonderzeichen:")

paths_with_special_characters = []

for path in image_folder.rglob("*"):
    try:
        str(path).encode("ascii")
    except UnicodeEncodeError:
        print(path)
        paths_with_special_characters.append(path)


# Tiefste Pfade zuerst bearbeiten.
# Dadurch werden zuerst Dateien und Unterordner umbenannt.
all_paths = sorted(
    image_folder.rglob("*"),
    key=lambda path: len(path.parts),
    reverse=True
)

renamed_count = 0

for old_path in all_paths:
    new_name = (
        old_path.name
        .replace("ß", "ss")
        .replace("ẞ", "SS")
    )

    # Der Name enthält kein ß
    if new_name == old_path.name:
        continue

    new_path = old_path.with_name(new_name)

    # Überschreiben vorhandener Dateien verhindern
    if new_path.exists():
        print(
            f"Nicht umbenannt, Ziel existiert bereits:\n"
            f"  Alt: {old_path}\n"
            f"  Neu: {new_path}"
        )
        continue

    old_path.rename(new_path)

    print(
        f"Umbenannt:\n"
        f"  Alt: {old_path.name}\n"
        f"  Neu: {new_name}"
    )

    renamed_count += 1


print(f"\nUmbenennungen abgeschlossen: {renamed_count}")


# Abschließend prüfen, ob noch ein ß vorhanden ist
remaining_paths = [
    path
    for path in image_folder.rglob("*")
    if "ß" in path.name or "ẞ" in path.name
]

if remaining_paths:
    print("\nFolgende Pfade enthalten weiterhin ein ß:")

    for path in remaining_paths:
        print(path)
else:
    print("Es wurden keine weiteren Pfade mit ß gefunden.")