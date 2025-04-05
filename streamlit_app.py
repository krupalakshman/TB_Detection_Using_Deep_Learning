import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os

# Page config
st.set_page_config(page_title="TB X-ray Detection", layout="wide")

# App title and description
st.title("Tuberculosis Detection from Chest X-rays")
st.write("Upload a chest X-ray image to detect the presence of Tuberculosis (TB) using deep learning models.")

# Upload file
uploaded_file = st.file_uploader("📤 Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

# Load models
@st.cache_resource()
def load_models():
    models = {}
    model_names = ["ResNet50", "VGG16", "EfficientNetB0", "MobileNetV2", "DenseNet121"]
    for name in model_names:
        path = os.path.join("models", f"{name}.keras")
        if os.path.exists(path):
            models[name] = tf.keras.models.load_model(path)
    return models

models = load_models()

# If image is uploaded
if uploaded_file:
    image = Image.open(uploaded_file)

    # Preprocess the image
    img = image.resize((256, 256))
    img_array = np.array(img)

    if len(img_array.shape) == 2:  # grayscale to RGB
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # remove alpha channel
        img_array = img_array[:, :, :3]

    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    results = []
    for model_name, model in models.items():
        pred_prob = model.predict(img_array)[0][0]
        prediction = "Tuberculosis" if pred_prob > 0.5 else "Normal"
        confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

        results.append({
            "Model": model_name,
            "Prediction": prediction,
            "Confidence (%)": round(confidence * 100, 2)
        })

    results_df = pd.DataFrame(results)

    # Layout: side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded X-ray")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔍 Model Predictions")
        st.dataframe(results_df, use_container_width=True)

    st.success("✅ Prediction complete. Always consult a medical professional for a verified diagnosis.")
