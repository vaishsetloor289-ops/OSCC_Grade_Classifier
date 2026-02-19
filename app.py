import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(
    page_title="OSCC Histopathology Classifier",
    layout="wide"
)

CLASS_MAP = {0: "Moderately Differentiated OSCC", 1: "Well Differentiated OSCC"}  # your mapping

# -----------------------
# HEADER
# -----------------------
st.markdown(
    """
    <div style="text-align:center; padding-top:10px;">
        <h1 style="margin-bottom:0;">🔬 Oral Squamous Cell Carcinoma - Histopathology Grading Classifier</h1>
        <h4 style="margin-top:8px; font-weight:500;">Built by Dr. Vaishnavi Setloor</h4>
        <p style="margin-top:6px; font-size:14px; opacity:0.9;">
            Deep Learning Model (ResNet-50)
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

# -----------------------
# MODEL LOADING
# -----------------------
@st.cache_resource
def load_model(weights_path: str):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 output classes

    state = torch.load(weights_path, map_location="cpu")

    # Handles both "state_dict only" and {"state_dict": ...} formats
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Sometimes keys have "module." prefix if trained with DataParallel
    new_state = {}
    for k, v in state.items():
        new_k = k.replace("module.", "")
        new_state[new_k] = v

    model.load_state_dict(new_state, strict=True)
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
        mean=[0.485, 0.456, 0.406],   # ImageNet norm
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(pil_img: Image.Image):
    img = pil_img.convert("RGB")
    x = infer_transform(img).unsqueeze(0)  # [1,3,224,224]
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [2]
        pred = int(np.argmax(probs))
    return pred, probs

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
            key=f"mag_{i}_{f.name}"   # ✅ unique key prevents crash
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

            # Show image + result
            col1, col2 = st.columns([1, 2], vertical_alignment="top")
            with col1:
                st.image(img, caption=f.name, use_container_width=True)
            with col2:
                st.markdown(f"**Prediction:** {label}")
                st.markdown(f"**Confidence:** {conf*100:.2f}%")
                st.markdown(f"**Magnification:** {magnifications.get(f.name, '-')}")
                st.progress(float(probs[pred]))

                # show both probabilities
                st.write("**Class Probabilities:**")
                st.write(f"- Moderate: {probs[0]*100:.2f}%")
                st.write(f"- Well: {probs[1]*100:.2f}%")

            st.write("---")

            per_image_rows.append({
                "Image": f.name,
                "Magnification": magnifications.get(f.name, None),
                "Prediction": label,
                "Moderate Prob (%)": round(float(probs[0]*100), 2),
                "Well Prob (%)": round(float(probs[1]*100), 2),
                "Confidence (%)": round(float(conf*100), 2),
            })

        # -----------------------
        # Aggregated Case Result
        # -----------------------
        st.subheader("Aggregated Case Result")

        # Majority voting
        final_pred = int(max(set(preds), key=preds.count))
        final_label = CLASS_MAP[final_pred]

        # Avg probability
        avg_probs = np.mean(np.stack(prob_list, axis=0), axis=0)  # [2]
        avg_conf = float(np.max(avg_probs))

        colA, colB = st.columns(2)
        with colA:
            st.markdown("### Majority Voting")
            st.success(f"Final Prediction: {final_label}")
            st.write(f"Agreement: {preds.count(final_pred)/len(preds)*100:.1f}%")
            st.write(f"Based on {len(preds)} images")

        with colB:
            st.markdown("### Average Probability")
            st.success(f"Final Prediction: {CLASS_MAP[int(np.argmax(avg_probs))]}")
            st.write(f"Avg Confidence: {avg_conf*100:.2f}%")
            st.write(f"- Moderate: {avg_probs[0]*100:.2f}%")
            st.write(f"- Well: {avg_probs[1]*100:.2f}%")

        # -----------------------
        # Table
        # -----------------------
        st.subheader("Detailed Results Table")
        df = pd.DataFrame(per_image_rows)
        st.dataframe(df, use_container_width=True)

# -----------------------
# FOOTER
# -----------------------
st.write("")
st.write("")
st.markdown(
    """
    <hr>
    <div style="text-align:center; font-size:13px; opacity:0.85;">
        This application is not intended for clinical diagnostic purposes. Only for research and academic purposes.
        <br>
        © 2026 Vaishnavi Setloor
    </div>
    """,
    unsafe_allow_html=True
)







