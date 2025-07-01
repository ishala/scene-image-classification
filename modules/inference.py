import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show_input_picture(sample_image, predicted_label):
    print(f"Prediksi Model    : {predicted_label}")

    plt.imshow(sample_image.squeeze(), cmap="gray")
    plt.title(f"Prediksi: {predicted_label}")
    plt.axis('off')
    plt.show()

def inference(imagePath, interpreter, input_details, output_details, class_names):
    # Memuat gambar dan mengonversinya ke format RGB
    image = Image.open(imagePath).convert("RGB")  # Mengonversi ke RGB (3 channel)

    image = image.resize((315, 315))

    sample_image = np.array(image) / 255.0  # Normalisasi gambar

    sample_image = sample_image.astype(np.float32)

    input_data = np.expand_dims(sample_image, axis=0)  # Menambahkan dimensi batch
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # Menjalankan model
    interpreter.invoke()

    # Mendapatkan output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)  # Menentukan index prediksi tertinggi
    predicted_label = class_names[predicted_index]  # Mengambil label sesuai dengan index

    show_input_picture(sample_image=sample_image, predicted_label=predicted_label)
    
    return predicted_label