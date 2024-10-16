import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

# Load TFLite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_with_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    return interpreter.get_tensor(output_details[0]['index'])

# Load Monkeypox vs Other Conditions model
model_monkeypox = load_tflite_model("monkeypox.tflite")

class_names_monkeypox = ['Monkeypox','Other Condition']

threshold_monkeypox = 0.5

def predict_image(img):
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred_monkeypox = predict_with_tflite(model_monkeypox, img_array.astype(np.float32))[0][0]
    
    label_monkeypox = class_names_monkeypox[int(pred_monkeypox > threshold_monkeypox)]
    conf_monkeypox = pred_monkeypox if pred_monkeypox > threshold_monkeypox else 1 - pred_monkeypox

    return label_monkeypox, conf_monkeypox

# Streamlit UI
st.set_page_config(page_title="Monkeypox Detector", page_icon="🦠", layout="wide")

# Custom CSS to adjust colors based on the poster
st.markdown(
    """
    <style>
    /* Background color for the main section */
    .main {
        background-color: #F1F8FF;
    }

    /* Title and headers */
    .stMarkdown h1, h2, h3 {
        color: #4B0082; /* Ungu tua */
    }

    /* Sidebar color */
    .css-1d391kg {
        background-color: #e3f2fd; /* Biru Muda */
    }

    /* Success box for correct classification */
    .st-success {
        background-color: #DFF2BF !important;
        color: #4B0082; /* Ungu */
    }

    /* Error box for wrong classification */
    .st-error {
        background-color: #FFBABA !important;
        color: #FF4500; /* Merah terang */
    }

    /* Button colors */
    .stButton>button {
        background-color: #FF4500; /* Merah terang */
        color: white;
    }

    /* Radio buttons and camera input label */
    .stRadio label, .stFileUploader label, .stCameraInput label {
        color: #4B0082; /* Ungu tua */
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("Monkeypox Detector 🦠")
st.write("Upload or capture an image to predict whether it shows signs of Monkeypox or other skin conditions.")

# Guidance for photo
st.write("**Guidance for Image Capture:**")
st.write("- Ensure the area is clean and visible without obstructions.")
st.write("- Focus on the skin area for accurate results.")

# Add a camera input option
camera_option = st.radio("Choose image source:", ("Upload Image", "Use Camera"))

uploaded_file = None
camera_image = None

if camera_option == "Upload Image":
    uploaded_file = st.file_uploader("Select an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = keras_image.load_img(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying image...")
elif camera_option == "Use Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption='Captured Image', use_column_width=True)
        st.write("Classifying image...")

if uploaded_file or camera_image:
    label_monkeypox, conf_monkeypox = predict_image(image)

    if label_monkeypox == 'Other Condition':
        st.error(f"Result: {label_monkeypox} ({conf_monkeypox:.2f})", icon="🚫")
    else:
        st.success(f"Result: {label_monkeypox} ({conf_monkeypox:.2f})", icon="🦠")

# Additional sections for better engagement
st.sidebar.header("General Information")
st.sidebar.write("Monkeypox is a viral disease that can be transmitted through close contact. Early detection is key to preventing the spread.")
st.sidebar.write("If symptoms are detected, it is advisable to consult a healthcare professional.")
