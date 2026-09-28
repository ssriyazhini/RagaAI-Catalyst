

        import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="AI Onion Quality Grading",
    page_icon="🧅"
)

st.title("🧅 AI Onion Quality Grading")
st.write("Capture or upload an onion image to check its quality.")


@st.cache_resource
def load_model():
    return YOLO("ragaai_catalyst/best.pt")


model = load_model()

camera_image = st.camera_input("📷 Capture Onion")

uploaded_image = st.file_uploader(
    "Or upload an onion image",
    type=["jpg", "jpeg", "png"]
)

image_file = camera_image if camera_image is not None else uploaded_image


if image_file is not None:

    image = Image.open(image_file).convert("RGB")

    st.image(
        image,
        caption="Selected Onion",
        use_container_width=True
    )

    if st.button("🔍 Analyze Onion"):

        results = model(
            image,
            conf=0.35,
            verbose=False
        )

        good_score = 0.0
        bad_score = 0.0

        for result in results:

            if result.boxes is None:
                continue

            for cls, conf in zip(
                result.boxes.cls,
                result.boxes.conf
            ):

                class_name = model.names[int(cls)]
                confidence = float(conf)

                if class_name == "GoodOnion":
                    good_score = max(good_score, confidence)

                elif class_name == "Bad_Onion":
                    bad_score = max(bad_score, confidence)


        # No onion detected
        if good_score == 0 and bad_score == 0:

            st.warning("⚠️ UNCERTAIN")

        # Good onion
        elif good_score > bad_score:

            st.success("✅ GOOD ONION")

        # Bad onion
        else:

            st.error("❌ BAD ONION")
