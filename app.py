
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Define the mapping from class index to character
# This mapping is based on the EMNIST ByClass dataset documentation
# https://www.nist.gov/system/files/documents/2017/04/24/emnist-mapping.txt
emnist_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i',
    45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r',
    54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'
}

# Define the path to the saved model file
model_path = 'emnist_byclass_model.h5'

# Load the trained model
@st.cache_resource
def load_model():
    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure 'emnist_byclass_model.h5' is in the same directory as app.py")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

st.title("AI Character Recognition")
st.write("Upload an image of a single character (0-9, A-Z, a-z) for recognition.")

# Ensure the model is loaded before allowing file upload
if model is None:
    st.stop() # Stop the app if model loading failed


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    try:
        # Read the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        # Convert to grayscale
        image = image.convert('L')
        # Resize to 28x28
        image = image.resize((28, 28))
        # Convert to numpy array and normalize
        image_array = np.array(image).astype('float32') / 255.0
        # Invert colors if the background is dark (EMNIST is black text on white background)
        # You might need to adjust this based on typical user uploads
        if np.mean(image_array) < 0.5:
             image_array = 1.0 - image_array
        # Reshape to (1, 28, 28, 1)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = np.expand_dims(image_array, axis=-1)


        # Make prediction
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_character = emnist_mapping.get(predicted_class_index, "Unknown")

        st.success(f"Predicted Character: **{predicted_character}**")

    except Exception as e:
        st.error(f"Error processing image or making prediction: {e}")

elif uploaded_file is None and model is None:
    st.warning("Model could not be loaded. Please check the model file path.")

