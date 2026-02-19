import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="OSCC Histopathology Grading Classifier",
    layout="wide"
)

# =========================
# CUSTOM STYLING
# =========================
st.markdown("""
<style>

/* Remove top spacing */
.block-container {
    padding-top: 2rem;
}

/* Full page oncology gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
}

/* White cards */
div[data-testid="stFileUploader"],
div[data-testid="stButton"],
div[data-testid="stNumberInput"],
div[data-testid="stDataFrame"] {
    background-color: rgba(255,255,255,0.95);
    padding: 15px;
    border-radius: 12px;
    color: black !important;
}

/* Make headers visible */
h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}

/* Header styling */
.header-title {
    font-size: 38px;
    font-weight: 800;
    text-align: center;
    color: #FFFFFF;
}

.subheader-text {
    font-size: 18px;
    text-align: center;
    color: #E0F2FF;
    margin-bottom: 30px;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 13px;
    color: #DDDDDD;
    margin-top: 60px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="header-title">
🔬 Oral Squamous Cell Carcinoma Histopathology Grading Classifier
</div>

<div class="subheader-text">
Deep Learning Model (ResNet-50) for Automated Differentiation Grading  
<br>
Built by Dr. Vaishnavi Setloor
</div>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load("oscc_resnet50_final.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class_names = [
    "Well Differentiated OSCC",
    "Moderately Differentiated OSCC",
    "Poorly Differentiated OSCC"
]

# =========================
# FILE UPLOADER
# =========================
st.subheader("Upload Histopathology Images")

uploaded_files = st.file_uploader(
    "Drag and drop files here",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:

    magnifications = {}

    st.subheader("🔬 Magnification Details")

    for i, file in enumerate(uploaded_files):
        mag = st.number_input(
            f"Magnification for {file.name}",
            min_value=1,
            max_value=100,
            value=10,
            key=f"{file.name}_{i}"
        )
        magnifications[file.name] = mag

    if st.button("🧠 Predict OSCC Grade"):

        predictions = []
        results_table = []

        for file in uploaded_files:

            image = Image.open(file).convert("RGB")
            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_class = torch.argmax(probs).item()

            predicted_label = class_names[pred_class]
            confidence = float(probs[pred_class])

            predictions.append(pred_class)

            # Show image
            st.image(image, caption=f"{file.name}", width=300)

            # Probability bars
            st.markdown("### Prediction Probabilities")
            for idx, cls in enumerate(class_names):
                st.progress(float(probs[idx]))
                st.write(f"{cls}: {probs[idx]*100:.2f}%")

            st.success(f"Predicted: {predicted_label} (Confidence: {confidence*100:.2f}%)")

            results_table.append({
                "Image Name": file.name,
                "Magnification": magnifications[file.name],
                "Predicted Grade": predicted_label,
                "Confidence (%)": round(confidence * 100, 2)
            })

            st.markdown("---")

        # =========================
        # RESULTS TABLE
        # =========================
        st.subheader("📊 Per-Image Results Summary")
        df = pd.DataFrame(results_table)
        st.dataframe(df, use_container_width=True)

        # =========================
        # AGGREGATED DECISION
        # =========================
        st.subheader("🧬 Aggregated Case-Level Prediction")

        final_pred = max(set(predictions), key=predictions.count)
        final_label = class_names[final_pred]

        st.markdown(f"### Final Predicted Grade: **{final_label}**")

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
This tool is intended strictly for research and academic purposes only.  
Not approved for clinical diagnostic decision-making.  
Developed using a ResNet-50 deep learning architecture trained on annotated histopathological images.
</div>
""", unsafe_allow_html=True)





