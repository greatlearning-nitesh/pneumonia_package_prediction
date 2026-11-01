import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "pneumonia_detection_project_package_pridiction_model_v1.joblib")
model = joblib.load(model_path)

st.title("Pneumonia Detection Prediction")
st.write("""
This application predicts whether a patient is likely to have pneumonia or not.
""")

# Input fields for the features
uploaded_image = st.file_uploader(
    "Upload a X-ray image",
    type=["jpg", "jpeg", "png"]
)

# Display uploaded image
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded X-ray image", use_container_width=True)

# Create a DataFrame from the input data
input_data = pd.DataFrame([{
   "image": uploaded_image
}])

if st.button("Predict Pneumonia"):
    # Predict probabilities and apply threshold
    prediction_proba = model.predict_proba(input_data)[:, 1]
    classification_threshold = 0.45 # Using the same threshold as in training
    prediction = (prediction_proba >= classification_threshold).astype(int)[0]

    result = "likely to have pneumonia" if prediction == 1 else "not likely to have pneumonia"
    st.subheader("Prediction Result:")
    st.success(f"The patient is **{result}** to have pneumonia.")
