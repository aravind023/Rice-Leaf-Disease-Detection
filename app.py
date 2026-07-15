import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Rice Leaf Disease Detection",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Rice Leaf Disease Detection")
st.markdown("### AI Powered Rice Disease Classification")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("best_rice_leaf_model.keras")
    except Exception as e:
        st.error(
            "⚠️ Model failed to load. This usually means the TensorFlow/Keras "
            "version installed here doesn't match the version the model was "
            "trained with. Check that requirements.txt pins the exact same "
            "tensorflow version used in the training notebook."
        )
        st.exception(e)
        st.stop()

model = load_model()

# -----------------------------
# Load Class Names
# -----------------------------
with open("class_names.json","r") as f:
    class_names = json.load(f)

# -----------------------------
# Disease Information
# -----------------------------
disease_info = {
    "Bacterial leaf blight":
        "Bacterial infection. Remove infected leaves and use recommended bactericides.",

    "Brown spot":
        "Fungal disease. Improve field drainage and apply suitable fungicide.",

    "Leaf smut":
        "Fungal disease. Use disease-free seeds and recommended fungicides."
}

uploaded_file = st.file_uploader(
    "Upload a Rice Leaf Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize((224,224))

    img = np.array(img)/255.0

    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    st.success(f"Prediction : {class_names[predicted_class]}")

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )

    st.info(disease_info[class_names[predicted_class]])

st.markdown("---")
st.caption("Developed by Aravindhan")
st.caption("🔗 [View source on GitHub](https://github.com/aravind023/Rice-Leaf-Disease-Detection)")
