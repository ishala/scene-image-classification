import streamlit as st
import tensorflow as tf
from PIL import Image
from modules.inference import inference

# Get Utils
with open("tflite/label.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load Model
interpreter = tf.lite.Interpreter(model_path="tflite/model.tflite")
interpreter.allocate_tensors()

# Get Detail Input and Output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("Scene Image Classifications")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Proses inference
    st.write("Make inference...")
    predicted_label = inference(uploaded_file, interpreter, input_details, output_details, class_names)

    st.subheader(predicted_label.capitalize())
