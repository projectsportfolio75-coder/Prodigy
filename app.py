import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

IMG_SIZE = 64
CATEGORIES = ['Cat', 'Dog']

# Load trained model
svm = joblib.load("svm_cat_dog_model.pkl")

st.title("🐶🐱 Cat vs Dog Classifier (SVM)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_flat = img.flatten().reshape(1, -1)

    prediction = svm.predict(img_flat)[0]
    st.subheader(f"Prediction: **{CATEGORIES[prediction]}**")
