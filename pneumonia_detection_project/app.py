import os
import numpy as np
import streamlit as st
import joblib
import pydicom
from pydicom.pixels import apply_voi_lut
from PIL import Image

# Image size and threshold used when the model was trained
IMG_SIZE = (100, 100)
CLASSIFICATION_THRESHOLD = 0.45

# Load the Keras model committed by the pipeline (sits next to this file).
# Cached so it is loaded once per container, not on every Streamlit rerun.
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "pneumonia_detection_project_package_pridiction_model_v1.joblib")
    return joblib.load(model_path)

def load_dicom(uploaded_file):
    # Convert a DICOM X-ray into an 8-bit grayscale PIL image
    ds = pydicom.dcmread(uploaded_file)
    pixels = ds.pixel_array
    if pixels.ndim == 3 and ds.get("SamplesPerPixel", 1) == 1:
        pixels = pixels[0]  # multi-frame: use the first frame
    pixels = apply_voi_lut(pixels, ds).astype("float32")  # apply the scanner's windowing
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        pixels = pixels.max() - pixels  # MONOCHROME1 stores bright as low values
    pixels -= pixels.min()
    if pixels.max() > 0:
        pixels /= pixels.max()
    return Image.fromarray((pixels * 255).astype("uint8")).convert("L")

def load_image(uploaded_file):
    if uploaded_file.name.lower().endswith(".dcm"):
        return load_dicom(uploaded_file)
    return Image.open(uploaded_file).convert("L")

def preprocess(image):
    # Model expects a batch of grayscale 100x100 images scaled to [0, 1]: shape (1, 100, 100, 1)
    array = np.asarray(image.resize(IMG_SIZE), dtype="float32") / 255.0
    return array.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

model = load_model()

st.title("Pneumonia Detection Prediction")
st.write("""
This application predicts whether a patient is likely to have pneumonia or not.
""")

uploaded_image = st.file_uploader(
    "Upload a X-ray image",
    type=["jpg", "jpeg", "png", "dcm"]
)

image = None
if uploaded_image is not None:
    try:
        image = load_image(uploaded_image)
        st.image(image, caption="Uploaded X-ray image")
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")

if st.button("Predict Pneumonia"):
    if image is None:
        st.warning("Please upload a valid X-ray image first.")
    else:
        prediction_proba = float(model.predict(preprocess(image), verbose=0)[0][0])
        prediction = int(prediction_proba >= CLASSIFICATION_THRESHOLD)

        result = "likely to have pneumonia" if prediction == 1 else "not likely to have pneumonia"
        st.subheader("Prediction Result:")
        st.success(f"The patient is **{result}** (probability {prediction_proba:.2f}).")
