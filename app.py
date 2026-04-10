import os
import streamlit as st
from PIL import Image

from inference import PneumoniaPredictor


DEFAULT_MODEL_PATH = os.path.join("models", "best_vit.pth")

st.set_page_config(page_title="Pneumonia Predictor", page_icon="🫁", layout="centered")

st.title("Chest X-ray Pneumonia Predictor")
st.write("Use Step 1 to load a trained model, then Step 2 to upload an X-ray image.")


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Model Settings")

    model_path = st.text_input("Model file path", value=DEFAULT_MODEL_PATH)
    uploaded_model = st.file_uploader("Upload model weights (.pth/.pt)", type=["pth", "pt"])

    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.01)


# =========================
# Model Handling
# =========================
effective_model_path = model_path

if uploaded_model is not None:
    os.makedirs("models", exist_ok=True)
    safe_name = os.path.basename(uploaded_model.name)
    effective_model_path = os.path.join("models", safe_name)

    with open(effective_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())

    st.sidebar.success(f"Using uploaded model: {safe_name}")


model_ready = os.path.exists(effective_model_path)

st.subheader("Step 1: Load Model")

if model_ready:
    st.success(f"Model found: {effective_model_path}")
else:
    st.warning(f"Model not found at: {effective_model_path}")
    st.info("Upload a model file or provide correct path.")


# =========================
# Load Predictor (SAFE)
# =========================
@st.cache_resource
def load_predictor(path: str, th: float):
    return PneumoniaPredictor(model_path=path, threshold=th)


predictor = None
torch_available = True

if model_ready:
    try:
        predictor = load_predictor(effective_model_path, threshold)
    except Exception as e:
        torch_available = False
        st.error(f"Model loading failed: {e}")
        st.info("⚠️ PyTorch is not supported on Streamlit Cloud (Python 3.14). Run locally or use API deployment.")


# =========================
# Image Upload & Prediction
# =========================
st.subheader("Step 2: Upload X-ray Image")

uploaded_file = st.file_uploader("Upload image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict", type="primary", disabled=(not model_ready or predictor is None)):
        try:
            result = predictor.predict(image)
            prob_percent = result["probability"] * 100

            st.subheader("Prediction")
            st.write(f"Pneumonia: {result['pneumonia']}")
            st.write(f"Confidence: {prob_percent:.2f}%")

            if result["pneumonia"] == "Yes":
                st.error("Model indicates pneumonia is present.")
            else:
                st.success("Model indicates no pneumonia.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif not model_ready:
    st.caption("Prediction is disabled until a valid model file is loaded.")


# =========================
# Footer
# =========================
st.markdown("---")
st.caption("⚠️ For educational/research use only. Not for clinical diagnosis.")

if not torch_available:
    st.warning("⚠️ Limited functionality: Model inference disabled in cloud environment.")