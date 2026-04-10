import os
import streamlit as st
from PIL import Image
import gdown

from inference import PneumoniaPredictor

DEFAULT_MODEL_PATH = os.path.join("models", "best_vit.pth")

st.set_page_config(page_title="Pneumonia Predictor", page_icon="🫁", layout="centered")

st.title("Chest X-ray Pneumonia Predictor")
st.write("Use Step 1 to load a trained model, then Step 2 to upload an X-ray image.")

# ✅ Auto download model from Google Drive
FILE_ID = "1bWmuk2GTV3LxQ6JUVs9OE5F36x_vBFZ7"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(DEFAULT_MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    with st.spinner("Downloading model from Google Drive... ⏳"):
        gdown.download(MODEL_URL, DEFAULT_MODEL_PATH, quiet=False)

with st.sidebar:
    st.header("Model Settings")
    model_path = st.text_input("Model file path", value=DEFAULT_MODEL_PATH)
    st.caption("Model uploader accepts only .pth or .pt files.")
    uploaded_model = st.file_uploader("Upload model weights", type=["pth", "pt"])
    threshold = st.slider("Decision threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.01)

effective_model_path = model_path

if uploaded_model is not None:
    os.makedirs("models", exist_ok=True)
    safe_name = os.path.basename(uploaded_model.name)
    effective_model_path = os.path.join("models", safe_name)
    with open(effective_model_path, "wb") as out_file:
        out_file.write(uploaded_model.getbuffer())
    st.sidebar.success(f"Using uploaded model: {effective_model_path}")

model_ready = os.path.exists(effective_model_path)

st.subheader("Step 1: Load Model")
if model_ready:
    st.success(f"Model found: {effective_model_path}")
else:
    st.warning(f"Model file not found at: {effective_model_path}")

@st.cache_resource
def load_predictor(path: str, th: float) -> PneumoniaPredictor:
    return PneumoniaPredictor(model_path=path, threshold=th)

predictor = None
if model_ready:
    try:
        predictor = load_predictor(effective_model_path, threshold)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        model_ready = False

st.subheader("Step 2: Upload X-ray Image")
uploaded_file = st.file_uploader("Upload image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict", type="primary", disabled=not model_ready):
        result = predictor.predict(image)
        prob_percent = result["probability"] * 100

        st.subheader("Prediction")
        st.write(f"Pneumonia: {result['pneumonia']}")
        st.write(f"Confidence: {prob_percent:.2f}%")

        if result["pneumonia"] == "Yes":
            st.error("Model indicates pneumonia is present.")
        else:
            st.success("Model indicates no pneumonia.")
elif not model_ready:
    st.caption("Prediction is disabled until a valid model file is loaded.")

st.caption("For educational/research use only. Not for clinical diagnosis.")