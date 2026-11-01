import sys
import os
import joblib
from datetime import datetime
import pandas as pd
from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError

label_names = {0: 'no pneumonia', 1: 'pneumonia'}

# Initialize Flask app
pneumonia_predictor_api = Flask("Pneumonia Detection Predictor")

# Load the trained model safely now
model_path = os.path.join(os.path.dirname(__file__), "../deployment/pneumonia_detection_project_package_pridiction_model_v1.joblib")
model = joblib.load(model_path)

# Define a route for the home page
@pneumonia_predictor_api.get('/')
def home():
    return "Welcome to the Pneumonia Detection Predictor API!"

# Define an endpoint to predict pneumonia detection
@pneumonia_predictor_api.post('/v1/predict')
def predict_pneumonia():
    # Get JSON data from the request
    try:
        # Check whether an image was uploaded
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image uploaded. Please upload an image.'
            }), 400

        uploaded_image = request.files['image']

        if uploaded_image.filename == '':
            return jsonify({
                'error': 'No image selected.'
            }), 400

        # Read and preprocess the image
        image = Image.open(
            io.BytesIO(uploaded_image.read())
        )

        # Convert image to grayscale
        image = image.convert('L')
        # Resize image
        image = image.resize(IMG_SIZE)

        # Convert image into NumPy array
        image_array = np.array(image, dtype=np.float32)

        # Normalize pixel values (must match training preprocessing)
        image_array = image_array / 255.0

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Generate prediction
        probability = float(model.predict(image_array, verbose=0)[0, 0])
        predicted_label = int(probability >= 0.5)
        predicted_class_name = label_names[predicted_label]
        # Extract predicted class
        predicted_class = np.asarray([predicted_label]).tolist()
    except UnidentifiedImageError:
        return jsonify({
            'error': 'Invalid image format. Please upload a valid image.'
        }), 400

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500
        
    return jsonify({
        'predicted_class': predicted_class,
        'predicted_class_name': predicted_class_name,
        'probability': probability
    }), 200

# Run the Flask app
if __name__ == '__main__':
    # FIX: Changed 'app.run' to match your defined variable name
    pneumonia_predictor_api.run(debug=True)
