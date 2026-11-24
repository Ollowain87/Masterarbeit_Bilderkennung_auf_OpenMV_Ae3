import numpy as np
import os
import tensorflow as tf
import tf_model as tf_model

model_dir = "my_saved_model"

model = tf_model.load_model()
model.export(model_dir)

image_dir = (
    r"C:\Users\nicoh\OneDrive\Desktop\Matlabcode_Teamprojekt"
    r"\09_Trainingsdaten\AusgeschnittenFürTrainingNeueVorverarbeitungMischDatenSatz"
)

imgSize = 48


def load_images_from_folder(
    folder,
    image_size=(imgSize, imgSize)
):
    images = []

    # Durchsucht den angegebenen Ordner und alle Unterordner.
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp")
            ):
                img_path = os.path.join(root, filename)

                image_bytes = tf.io.read_file(img_path)

                img = tf.io.decode_image(
                    image_bytes,
                    channels=3,
                    expand_animations=False
                )

                img.set_shape([None, None, 3])

                img = tf.image.resize(
                    img,
                    size=image_size,
                    method=tf.image.ResizeMethod.BILINEAR,
                    antialias=True
                )

                img = img.numpy().astype(np.float32)

                print("Bild geladen mit Shape:", img.shape)
                images.append(img)

    return images


dataset = load_images_from_folder(image_dir)

print(f"Insgesamt geladene Bilder: {len(dataset)}")


def representative_dataset():
    for image in dataset:
        image = np.expand_dims(image, axis=0)

        if image.shape != (1, imgSize, imgSize, 3):
            print("❌ Falsche Bildform:", image.shape)
            continue

        yield [image.astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = representative_dataset

converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open("sign-classifier_hybrid.tflite", "wb") as f:
    f.write(tflite_model)

print(
    "Hybrides TensorFlow-Lite-Modell wurde "
    "erfolgreich gespeichert!"
)