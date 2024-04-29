import os
import json
from PIL import Image
import base64
import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"/home/garv/Downloads/demeter/app/model.h5"

model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Set the title with black text color and increased font size using Markdown
st.markdown("<h1 style='color: black; font-size: 46px; margin-bottom: 50px;'>DEMETER: Plant Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color: black;'>Guidelines</h3>", unsafe_allow_html=True)
# Define the guideline text with a dark box background and white text color
guideline_text = """
<div style='background-color: #262730; padding: 10px; border-radius: 10px; margin-bottom: 50px;margin-top: -10px;'>
    <span style='color: white;'>Our model accuracy is 98%. Demeter's disease diagnosis and treatment recommendations are based on a deep learning model and provided as guidance only. No technology is 100% accurate in all situations.
    <br><br>
    Upload an image of the plant's leaf in natural or good lighting to get better and accurate results.</span>
</div>
"""
st.markdown(guideline_text, unsafe_allow_html=True)




uploaded_image = st.file_uploader("Upload an image...")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Upload'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(image_url):
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("%s");
    background-size: cover;
    }
    </style>
    ''' % image_url
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the function to set the background image
set_background('https://facts.net/wp-content/uploads/2023/09/14-fascinating-facts-about-plant-reproduction-1693828613.jpg')