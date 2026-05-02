from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model


MODEL_PATH = Path("models/best_inceptionv3_model.keras")
IMAGE_SIZE = (299, 299)
CLASS_NAMES = ["abnormal", "normal"]


st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    layout="centered",
)


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def normalize_probabilities(raw_prediction: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(raw_prediction, dtype=np.float32).reshape(-1)

    if probabilities.size == 1:
        dr_probability = float(probabilities[0])
        return np.array([dr_probability, 1.0 - dr_probability], dtype=np.float32)

    probabilities = probabilities[: len(CLASS_NAMES)]
    total = float(probabilities.sum())
    if total > 0:
        probabilities = probabilities / total
    return probabilities


st.title("Diabetic Retinopathy Detection")
st.caption("Upload a retinal image to classify it as abnormal or normal.")

st.warning(
    "This app is for educational screening support only. It is not a medical diagnosis."
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info(
        "Export your trained model from Colab, place it at "
        "`models/best_inceptionv3_model.keras`, then redeploy the app."
    )
    st.stop()

model = get_model()

uploaded_file = st.file_uploader(
    "Upload retinal image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.stop()

image = Image.open(uploaded_file)
st.image(image, caption="Uploaded image", use_container_width=True)

if st.button("Predict", type="primary", use_container_width=True):
    with st.spinner("Analyzing image..."):
        model_input = preprocess_image(image)
        raw_prediction = model.predict(model_input, verbose=0)
        probabilities = normalize_probabilities(raw_prediction)

    predicted_index = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    st.subheader("Prediction")
    st.metric(predicted_label.title(), f"{confidence:.2%}")

    st.progress(confidence)

    st.subheader("Class Scores")
    for label, probability in zip(CLASS_NAMES, probabilities):
        st.write(f"{label.title()}: {float(probability):.2%}")
