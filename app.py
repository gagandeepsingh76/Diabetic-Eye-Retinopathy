from PIL import Image, ImageOps, ImageStat
import streamlit as st


st.set_page_config(
    page_title="Diabetic Retinopathy Project",
    layout="centered",
)

st.title("Diabetic Retinopathy Project")
st.caption("Upload a retinal image to preview basic image details inside the web app.")

st.warning(
    "This app is for educational project display only. It does not provide a medical diagnosis."
)

st.info(
    "This deployment demonstrates the image upload and preprocessing interface for the project."
)

uploaded_file = st.file_uploader(
    "Upload retinal image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.stop()

image = Image.open(uploaded_file)
image = ImageOps.exif_transpose(image).convert("RGB")

st.image(image, caption="Uploaded retinal image", use_container_width=True)

width, height = image.size
brightness = sum(ImageStat.Stat(image).mean) / 3

st.subheader("Image Preview Details")
st.write(f"Image size: {width} x {height} pixels")
st.write(f"Average brightness: {brightness:.1f} / 255")

st.success("Image uploaded successfully.")
