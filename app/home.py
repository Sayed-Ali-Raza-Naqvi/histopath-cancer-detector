import streamlit as st
import requests
import base64
import io
from PIL import Image


st.set_page_config(
    page_title="Histopathologoy Cancer Detector",
    page_icon=":microscope:",
    layout="wide"
)

st.sidebar.title(":microscope: Histopathologoy Cancer Detector")

st.sidebar.markdown(
    """
    An explainable deep learning system for detecting 
    metastatic cancer in histopathology tissue patches.
    
    **Model:** EfficientNet-B0  
    **Dataset:** PatchCamelyon (327,680 patches)  
    **Explainability:** Grad-CAM heatmaps  
    """
)

st.sidebar.divider()

st.sidebar.info(
    "⚠️ For research use only. "
    "Not a clinical diagnostic tool."
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **How to use:**
    1. Upload a histopathology patch (PNG/JPEG)
    2. Click **Analyze**
    3. View prediction and Grad-CAM heatmap
    
    **Navigation:**
    - Home — upload and predict
    - Model Info — metrics and plots
    """
)

API_URL = "http://localhost:8000/api/v1"

def decode_base64_image(b64_string: str) -> Image.Image:
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes))


def call_predict_api(image_bytes: bytes, filename: str, content_type: str):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": (filename, image_bytes, content_type)},
            timeout=60
        )
        return response
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.Timeout:
        return None
    

st.title("Cancer Detection from Histopathology Patches")

st.markdown(
    "Upload a tissue patch image to classify it as **tumor** or **normal** "
    "with an explainable Grad-CAM heatmap."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload a histopathology patch image (JPEG or PNG format only)",
    type=["jpg", "jpeg", "png"],
    help="Recommended: 96×96px H&E stained tissue patch."
)

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    content_type = "image/png" if uploaded_file.name.lower().endswith("png") else "image/jpeg"

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Uploaded Patch")
        st.image(image, width=250)
        st.caption(f"Size: {image.size[0]}×{image.size[1]} px")
    
    with col2:
        st.subheader("Analysis")
        analyze = st.button("Analyze Patch", type="primary")

        if analyze:
            with st.spinner("Running inference and generating Grad-CAM..."):
                response = call_predict_api(image_bytes, uploaded_file.name, content_type)

            if response is None:
                st.error(
                    "Could not connect to the API."
                    "Make sure the server is running on port 8000."
                )
            elif response.status_code != 200:
                detail = response.json().get("detail", "Unknown error")
                st.error(f"API error {response.status_code}: {detail}")
            else:
                data = response.json()
                prediction = data["prediction"]
                tumor_prob = data["tumor_probability"]
                confidence = data["confidence"]
                threshold = data["threshold_used"]

                if prediction == "Tumor":
                    st.error(f"Prediction: **{prediction}**")
                else:
                    st.success(f"Prediction: **{prediction}**")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Tumor Probability", f"{tumor_prob:.2%}")
                m2.metric("Confidence", f"{confidence:.2%}")
                m3.metric("Threshold", f"{threshold:.2%}")
                
                if prediction == "Tumor":
                    bar_color = "red" if tumor_prob > 0.7 else "orange"
                else:
                    bar_color = "green"
                
                st.progress(
                    tumor_prob,
                    text=f"Tumor Probability: {tumor_prob:.2%}",
                )

                st.divider()

                st.subheader("Grad-CAM Heatmap")
                st.caption(
                    "Red/yellow regions indicate where the model focused. "
                    "Blue regions had low influence on the prediction."
                )

                gradcam_image = decode_base64_image(data["gradcam_image"])

                c1, c2 = st.columns(2)

                with c1:
                    st.image(image, caption="Original Patch", width=250)
                with c2:
                    st.image(gradcam_image, caption="Grad-CAM Heatmap", width=250)
                
                st.divider()

                with st.expander("Raw API Response"):
                    display_data = {k: v for k, v in data.items() if k != "gradcam_image"}
                    st.json(display_data)
                
                st.caption(
                    "⚠️ " + data.get(
                        "disclaimer",
                        "For research use only. Not a clinical diagnostic tool."
                    )
                )