import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="OSCC Grade Classifier",
    page_icon="🔬",
    layout="wide"
)

# =========================================
# CLINICAL UI STYLING
# =========================================

st.markdown("""
<style>

.main {
    background-color: #F4F8FB;
}

h1 {
    color: #1F4E79;
}

.pred-box {
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}

.moderate {
    background-color: #FFF3CD;
    border-left: 8px solid #FFB000;
}

.well {
    background-color: #E6F4EA;
    border-left: 8px solid #2E7D32;
}

.footer {
    text-align: center;
    color: grey;
    margin-top: 50px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODEL
# =========================================

@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("oscc_resnet50_final.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# =========================================
# TRANSFORMS
# =========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class_names = ["Moderate", "Well"]

# =========================================
# HEADER
# =========================================

st.title("🔬 OSCC Grade Classifier")
st.markdown("AI-assisted histopathological grading of Oral Squamous Cell Carcinoma.")
st.markdown("---")

# =========================================
# FILE UPLOAD
# =========================================

uploaded_files = st.file_uploader(
    "Upload Histopathology Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# =========================================
# PROCESS IF FILES EXIST
# =========================================

if uploaded_files:

    magnifications = {}

    st.subheader("🔎 Magnification Details")

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
        per_image_results = []

        for file in uploaded_files:

            image = Image.open(file).convert("RGB")
            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()

            predictions.append(pred)
            probabilities.append(probs.numpy()[0])

            per_image_results.append({
                "Image Name": file.name,
                "Magnification": magnifications[file.name],
                "Prediction": class_names[pred],
                "Moderate Probability (%)": round(probs.numpy()[0][0] * 100, 2),
                "Well Probability (%)": round(probs.numpy()[0][1] * 100, 2)
            })

        # =========================================
        # AGGREGATION
        # =========================================

        predictions = np.array(predictions)
        probabilities = np.array(probabilities)

        majority_vote = np.bincount(predictions).argmax()
        avg_probs = probabilities.mean(axis=0)

        st.markdown("---")
        st.subheader("📊 Case-Level Prediction")

        if class_names[majority_vote] == "Moderate":
            st.markdown(f"""
            <div class="pred-box moderate">
            <h3>Final Prediction: Moderate Differentiated OSCC</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-box well">
            <h3>Final Prediction: Well Differentiated OSCC</h3>
            </div>
            """, unsafe_allow_html=True)

        # =========================================
        # PROBABILITY BARS
        # =========================================

        st.write("### Average Case Probability")

        st.progress(float(avg_probs[0]))
        st.write(f"Moderate: {avg_probs[0]*100:.2f}%")

        st.progress(float(avg_probs[1]))
        st.write(f"Well: {avg_probs[1]*100:.2f}%")

        # =========================================
        # PER IMAGE TABLE
        # =========================================

        st.markdown("---")
        st.subheader("📁 Per-Image Results")

        df = pd.DataFrame(per_image_results)
        st.dataframe(df, use_container_width=True)

# =========================================
# FOOTER
# =========================================

st.markdown("---")
st.markdown('<div class="footer">Built by Dr. Vaishnavi Setloor</div>', unsafe_allow_html=True)



