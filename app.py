# app.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO

# --- Configuration & Model Loading ---

# Use st.cache_resource for efficient model loading (loads only once)
@st.cache_resource
def load_tf_model():
    """Loads the pre-trained Keras model."""
    try:
        model = tf.keras.models.load_model("./trained_model/traffic_classifier_v4.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Dictionary to map class indices to their names
CLASSES = {
    0: 'Speed Limit 20 km/h', 1: 'Speed Limit 30 km/h', 2: 'Speed Limit 50 km/h', 3: 'Speed Limit 60 km/h',
    4: 'Speed Limit 70 km/h', 5: 'Speed Limit 80 km/h', 6: 'End of Speed Limit 80 km/h', 7: 'Speed Limit 100 km/h',
    8: 'Speed Limit 120 km/h', 9: 'No Overtaking', 10: 'No Overtaking for Trucks', 11: 'Crossroads Ahead',
    12: 'Priority Road', 13: 'Give Way', 14: 'Stop', 15: 'No Vehicles',
    16: 'Vehicles over 3.5 tons prohibited', 17: 'No Entry', 18: 'General Caution',
    19: 'Dangerous Curve to the Left', 20: 'Dangerous Curve to the Right', 21: 'Double Bend First to the Left',
    22: 'Uneven Road', 23: 'Slippery Road', 24: 'Road Narrows on the Right', 25: 'Road Work',
    26: 'Traffic Signals', 27: 'Pedestrians Crossing', 28: 'Children Crossing', 29: 'Bicycles Crossing',
    30: 'Beware of Ice/Snow', 31: 'Wild Animals Crossing', 32: 'End of All Speed and Passing Limits',
    33: 'Turn Right Ahead', 34: 'Turn Left Ahead', 35: 'Ahead Only', 36: 'Go Straight or Right',
    37: 'Go Straight or Left', 38: 'Keep Right', 39: 'Keep Left', 40: 'Roundabout Mandatory',
    41: 'End of No Passing', 42: 'End of No Passing by Trucks',
}

# --- Image Processing & Prediction Functions ---

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocesses the image for the model."""
    img = image.resize((30, 30))
    img = np.array(img)
    # Ensure image has 3 channels (RGB)
    if img.ndim == 2:  # Grayscale
        img = np.stack((img,) * 3, axis=-1)
    if img.shape[-1] == 4:  # RGBA
        img = img[:, :, :3]
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def predict(model: tf.keras.Model, image: Image.Image) -> tuple[int, float]:
    """Makes a prediction using the loaded model."""
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    return class_index, confidence

# --- Streamlit User Interface ---

# Set page configuration
st.set_page_config(page_title="Traffic Sign Classifier", page_icon="🚦", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B4B4B;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #808080;
    }
    .result-box {
        background-color: #F0F2F6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .result-text {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .confidence-text {
        font-size: 1.1rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<p class="main-header">Traffic Sign Classifier 🚦</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">An AI-powered application to identify traffic signs from your images.</p>', unsafe_allow_html=True)
st.markdown("---")

# Load the model
model = load_tf_model()

if model:
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a traffic sign image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        # Initial state message
        st.info("👋 Welcome! Please upload an image to get started.")
    else:
        # Process the uploaded file
        image = Image.open(uploaded_file)
        
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            # Get prediction
            class_index, confidence = predict(model, image)
            class_name = CLASSES.get(class_index, "Unknown Sign")

            # Display the result in a styled box
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown('<h3>Prediction Result</h3>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-text">{class_name}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="confidence-text">Confidence: {confidence*100:.2f}%</p>', unsafe_allow_html=True)
            st.progress(confidence)
            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.error("Model could not be loaded. Please ensure the model file is in the correct directory.")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: grey;">'
    'Project Created By: Youssef Mustafa Hussein'
    '</div>',
    unsafe_allow_html=True
)
