import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="OSCC Grade Classifier",
    page_icon="🔬",
    layout="wide"
)

# =========================
# CUSTOM STYLING
# =========================
st.markdown("""
<style>
.main {
    background-color: #F4F8FB;
}

.title-text {
    font-size: 36px;
    font-weight: 700;
    color: #1B4F72;
    text-align: center;
}

.subtitle-text {
    font-size: 18px;
    color: #566573;
    text-align: center;
}

.prediction-card {
    padding: 25px;
    border-radius: 12px;
    font-size: 24px;
    font-weight: 600;
    text-align: center;
}

.footer {
    text-align: center;
    font-size: 14px;
    color: grey;
    margin-top: 60px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<div class="title-text">🔬 Oral Squamous Cell Carcinoma (OSCC) Grade Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">AI-based histopathology grading using Fine-Tuned ResNet50</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Built by <b>Dr. Vaishnavi Setloor</b></div>', unsafe_allow_html=True)

st.divider()

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("oscc_resnet50_final.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class_names = ["Moderately Differentiated", "Well Differentiated"]

# =========================
# IMAGE UPLOAD
# =========================
uploaded_files = st.file_uploader(
    "Upload Histopathology Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:

    magnifications = {}

    st.subheader("🔍 Magnification Details")
    for i, file in enumerate(uploaded_files):
    mag = st.number_input(
        f"Magnification for {file.name}",
        min_value=1,
        max_value=100,
        value=10,
        key=f"{file.name}_{i}"
    )
    magnifications[file.name] = mag

    if st.button("🔎 Predict OSCC Grade"):

        predictions = []
        probabilities = []

        for file in uploaded_files:
            image = Image.open(file).convert("RGB")
            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()

            predictions.append(pred)
            probabilities.append(probs.numpy()[0])

        # =========================
        # AGGREGATION
        # =========================
        final_prediction = max(set(predictions), key=predictions.count)
        avg_probs = np.mean(probabilities, axis=0)

        st.divider()
        st.subheader("📊 Aggregated Case Result")

        if final_prediction == 0:
            bg_color = "#FDEBD0"   # soft orange
        else:
            bg_color = "#D6EAF8"   # soft blue

        st.markdown(
            f"""
            <div class="prediction-card" style="background-color:{bg_color};">
            Final Prediction: {class_names[final_prediction]}
            </div>
            """,
            unsafe_allow_html=True
        )

        confidence_score = avg_probs[final_prediction]

        st.write(f"### Confidence: {confidence_score*100:.2f}%")
        st.progress(float(confidence_score))

        st.write("### Average Class Probabilities")
        st.write(f"Moderate: {avg_probs[0]*100:.2f}%")
        st.write(f"Well: {avg_probs[1]*100:.2f}%")

        # =========================
        # INDIVIDUAL RESULTS
        # =========================
        st.divider()
        st.subheader("📋 Individual Image Predictions")

        for i, file in enumerate(uploaded_files):
            st.write(f"**{file.name}**")
            st.write(f"Prediction: {class_names[predictions[i]]}")
            st.write(f"Moderate Probability: {probabilities[i][0]*100:.2f}%")
            st.write(f"Well Probability: {probabilities[i][1]*100:.2f}%")
            st.write("---")

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
This AI tool is intended for research and educational purposes only.<br>
© 2026 Dr. Vaishnavi Setloor
</div>
""", unsafe_allow_html=True)


