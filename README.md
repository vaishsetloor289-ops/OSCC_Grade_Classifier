# OSCC Grade Classifier

Deep learning model that grades oral squamous cell carcinoma (OSCC) differentiation as well, moderately or poorly differentiated from H&E histopathology images.

Built by Dr Vaishnavi Setloor 
## Live demo
[Try the app]https://osccgradeclassifier.streamlit.app

## Method
- ImageNet-pretrained ResNet-50, fine-tuned in PyTorch
- 320 H&E image patches at mixed magnifications (4×, 20×, 40×)
- Ground truth from two oral pathologists with consensus review
- Case-level train/validation/test split to prevent data leakage


## Results (test set, 46 images)
- Accuracy: 84.8%
- Macro F1: 0.82
- Per-class AUC: 0.85–0.98

## Disclaimer
Research prototype. Not for clinical diagnosis.
