import os
import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Onion Grading AI", page_icon="🧅")
st.title("🧅 Onion Grading AI")
st.write("Upload or capture an onion image to check its quality.")

MODEL_URL = "https://github.com/ssriyazhini/RagaAI-Catalyst/releases/download/v1.0/best.4.pt"
MODEL_PATH = "/tmp/best.pt"

if not os.path.exists(MODEL_PATH): st.info("Loading AI model..."); open(MODEL_PATH, "wb").write(requests.get(MODEL_URL, timeout=120).content)

model = YOLO(MODEL_PATH)
st.success("AI model loaded successfully!")

camera_image = st.camera_input("📷 Capture Onion")
uploaded_image = st.file_uploader("Or upload an onion image", type=["jpg", "jpeg", "png"])

image_file = camera_image if camera_image is not None else uploaded_image

if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Selected Onion", use_container_width=True)
    if st.button("🔍 Check Onion Quality"):
        results = model.predict(source=image, conf=0.25, verbose=False)
        result = results[0]
        detected = len(result.boxes) > 0
        if detected:
            best_index = result.boxes.conf.argmax()
            best_class = int(result.boxes.cls[best_index])
            class_name = result.names[best_class].lower()
            if class_name == "good": st.success("🟢 GOOD ONION")
            else: st.error("🔴 BAD ONION")
        else: st.warning("⚠️ Onion could not be detected. Please try another image.")
