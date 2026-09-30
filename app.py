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

if not os.path.exists(MODEL_PATH):
st.info("Loading AI model...")
response = requests.get(MODEL_URL, timeout=120)
response.raise_for_status()

```
with open(MODEL_PATH, "wb") as f:
    f.write(response.content)
```

model = YOLO(MODEL_PATH)

st.success("AI model loaded successfully!")

camera_image = st.camera_input("📷 Capture Onion")

uploaded_image = st.file_uploader(
"Or upload an onion image",
type=["jpg", "jpeg", "png"]
)

if camera_image is not None:
image_file = camera_image
elif uploaded_image is not None:
image_file = uploaded_image
else:
image_file = None

if image_file is not None:
image = Image.open(image_file)

```
st.image(
    image,
    caption="Selected Onion",
    use_container_width=True
)

if st.button("🔍 Check Onion Quality"):
    st.write("Analyzing...")

    results = model.predict(
        source=image,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    if result.boxes is not None and len(result.boxes) > 0:
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        best_index = confidences.argmax()
        best_class = int(class_ids[best_index])

        class_name = result.names[best_class].lower()

        if class_name == "good":
            st.success("🟢 GOOD ONION")
        else:
            st.error("🔴 BAD ONION")
    else:
        st.warning("⚠️ Onion could not be detected. Please try another image.")
```

