import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from collections import Counter
from textwrap import dedent

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(
    page_title="OSCC Histopathology Grading Classifier",
    layout="wide"
)

CLASS_MAP = {
    0: "Moderately Differentiated OSCC",
    1: "Well Differentiated OSCC"
}

# -----------------------
# OPTIONAL PAGE STYLING
# -----------------------
st.markdown(
    dedent("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .thesis-header {
        text-align: center;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        line-height: 1.5;
    }
    .thesis-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1f2937;
    }
    .thesis-subtitle {
        font-size: 1.2rem;
        font-weight: 500;
        margin-top: 0.4rem;
        margin-bottom: 0.2rem;
        color: #374151;
    }
    .thesis-text {
        font-size: 1rem;
        margin-top: 0.15rem;
        margin-bottom: 0.15rem;
        color: #4b5563;
    }
    .ack-box {
        margin-top: 0.9rem;
        margin-bottom: 0.6rem;
        font-size: 0.98rem;
        color: #374151;
    }
    .model-note {
        font-size: 0.9rem;
        opacity: 0.85;
        margin-top: 0.5rem;
    }
    .summary-card {
        background-color: #f8fafc;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .small-muted {
        font-size: 0.92rem;
        color: #6b7280;
    }
    </style>
    """),
    unsafe_allow_html=True
)

# -----------------------
# HEADER
# -----------------------
st.markdown(
    dedent("""
    <div class="thesis-header">
        <div class="thesis-title">
            🔬 Oral Squamous Cell Carcinoma - Histopathology Grading Classifier
        </div>

        <div class="thesis-subtitle">
            Built by Dr. Vaishnavi Setloor
        </div>

        <div class="thesis-text">
            Under the guidance of Dr. Sahana Srinath, Professor &amp; Head, Department of Oral Pathology
        </div>

        <div class="thesis-text">
            Government Dental College &amp; Research Institute, Bengaluru
        </div>

        <div class="ack-box">
            <strong>Acknowledgements</strong><br>
            Dr. Satish Yadav<br>
            Dr. Jyoti Tahasildhar
        </div>

        <div class="model-note">
            Deep Learning Model (ResNet-50)
        </div>
    </div>
    """),
    unsafe_allow_html=True
)

st.write("---")

# -----------------------
# MODEL LOADING
# -----------------------
@st.cache_resource
def load_model(weights_path: str):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state = torch.load(weights_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned_state = {}
    for k, v in state.items():
        cleaned_state[k.replace("module.", "")] = v

    model.load_state_dict(cleaned_state, strict=True)
    model.eval()
    return model

MODEL_PATH = "oscc_resnet50_final.pth"
model = load_model(MODEL_PATH)

# -----------------------
# TRANSFORMS (INFERENCE)
# -----------------------
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(pil_img: Image.Image):
    img = pil_img.convert("RGB")
    x = infer_transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    return pred, probs

# -----------------------
# INTRO NOTE
# -----------------------
st.markdown(
    dedent("""
    <div class="small-muted">
        Upload one or multiple histopathology images in JPG, JPEG, or PNG format.
        You may enter the corresponding magnification for each uploaded image before running prediction.
    </div>
    """),
    unsafe_allow_html=True
)

st.write("")

# -----------------------
# UI: Upload
# -----------------------
st.subheader("Upload Histopathology Images")

uploaded_files = st.file_uploader(
    "Upload one or multiple images (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("Magnification Details")

    magnifications = {}
    for i, f in enumerate(uploaded_files):
        mag = st.number_input(
            f"Magnification for: {f.name}",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key=f"mag_{i}_{f.name}"
        )
        magnifications[f.name] = mag

    st.write("")

    if st.button("Predict"):
        per_image_rows = []
        preds = []
        prob_list = []

        st.subheader("Individual Image Results")

        for i, f in enumerate(uploaded_files):
            img = Image.open(f)
            pred, probs = predict_image(img)

            preds.append(pred)
            prob_list.append(probs)

            label = CLASS_MAP[pred]
            conf = float(probs[pred])

            col1, col2 = st.columns([1, 2], vertical_alignment="top")

            with col1:
                st.image(img, caption=f.name, use_container_width=True)

            with col2:
                st.markdown(f"**Prediction:** {label}")
                st.markdown(f"**Confidence:** {conf * 100:.2f}%")
                st.markdown(f"**Magnification:** {magnifications.get(f.name, '-')}")
                st.progress(float(probs[pred]))
                st.write("**Class Probabilities:**")
                st.write(f"- Moderately Differentiated OSCC: {probs[0] * 100:.2f}%")
                st.write(f"- Well Differentiated OSCC: {probs[1] * 100:.2f}%")

            st.write("---")

            per_image_rows.append({
                "Image": f.name,
                "Magnification": magnifications.get(f.name, None),
                "Prediction": label,
                "Moderately Differentiated OSCC Prob (%)": round(float(probs[0] * 100), 2),
                "Well Differentiated OSCC Prob (%)": round(float(probs[1] * 100), 2),
                "Confidence (%)": round(float(conf * 100), 2),
            })

        st.subheader("Aggregated Case Result")

        vote_counts = Counter(preds)
        final_pred_majority = vote_counts.most_common(1)[0][0]
        final_label_majority = CLASS_MAP[final_pred_majority]
        agreement = (vote_counts[final_pred_majority] / len(preds)) * 100

        avg_probs = np.mean(np.stack(prob_list, axis=0), axis=0)
        final_pred_avg = int(np.argmax(avg_probs))
        final_label_avg = CLASS_MAP[final_pred_avg]
        avg_conf = float(np.max(avg_probs))

        st.markdown(
            dedent(f"""
            <div class="summary-card">
                <h4 style="margin-top:0; margin-bottom:0.6rem;">Case Summary</h4>
                <p style="margin:0.2rem 0;"><strong>Total Images Analysed:</strong> {len(preds)}</p>
                <p style="margin:0.2rem 0;"><strong>Majority Vote Prediction:</strong> {final_label_majority}</p>
                <p style="margin:0.2rem 0;"><strong>Average Probability Prediction:</strong> {final_label_avg}</p>
                <p style="margin:0.2rem 0;"><strong>Majority Agreement:</strong> {agreement:.1f}%</p>
                <p style="margin:0.2rem 0;"><strong>Average Confidence:</strong> {avg_conf * 100:.2f}%</p>
            </div>
            """),
            unsafe_allow_html=True
        )

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### Majority Voting Result")
            st.success(f"Final Prediction: {final_label_majority}")
            st.write(f"Agreement across uploaded images: {agreement:.1f}%")
            st.write(f"Based on {len(preds)} uploaded image(s)")

        with colB:
            st.markdown("### Average Probability Result")
            st.success(f"Final Prediction: {final_label_avg}")
            st.write(f"Average confidence: {avg_conf * 100:.2f}%")
            st.write(f"- Moderately Differentiated OSCC: {avg_probs[0] * 100:.2f}%")
            st.write(f"- Well Differentiated OSCC: {avg_probs[1] * 100:.2f}%")

        st.subheader("Detailed Results Table")
        df = pd.DataFrame(per_image_rows)
        st.dataframe(df, use_container_width=True)

# -----------------------
# FOOTER
# -----------------------
st.write("")
st.write("")

st.markdown(
    dedent("""
    <hr>
    <div style="text-align:center; font-size:13px; opacity:0.85; line-height:1.6;">
        This application is intended strictly for research and academic purposes and not for clinical diagnostic use.
        <br>
        Developed as part of thesis-related academic work.
        <br>
        © 2026 Vaishnavi Setloor
    </div>
    """),
    unsafe_allow_html=True
)





