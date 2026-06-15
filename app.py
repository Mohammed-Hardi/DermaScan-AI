"""Streamlit web application."""

import streamlit as st
import torch
import json
from PIL import Image
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from predict import DermaScanPredictor
from config import *

st.set_page_config(page_title="DermaScan AI", layout="wide")

@st.cache_resource
def load_predictor():
    return DermaScanPredictor()


def main():
    st.title("🩺 DermaScan AI")
    st.subheader("5-Class Skin Lesion Classification")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info("AI-powered skin lesion screening using deep learning.")
    st.sidebar.markdown("**Detects:**")
    for cls in SELECTED_CLASSES:
        st.sidebar.markdown(f"- {CLASS_NAMES[cls]}")
    
    # Check if model is trained / well-trained
    model_path = Path("models/best_model.pth")
    status_path = Path("models/training_status.json")
    if not model_path.exists():
        st.info("📋 **Model Status**: Untrained")
        st.markdown("""
        To train the model, open a terminal and run:
        ```bash
        python src/train.py
        ```
        
        The model will take ~30-60 minutes to train on CPU.
        Once trained, refresh this page and upload images to test.
        """)
    elif status_path.exists():
        with open(status_path) as f:
            status = json.load(f)
        if not status.get("is_well_trained", True):
            st.warning(
                f"⚠️ **Model Status**: Checkpoint exists but best validation accuracy was only "
                f"**{status.get('best_val_acc', '?')}%** "
                f"(minimum required: {status.get('min_required_acc', 30)}%). "
                "Continue training or check your dataset before relying on predictions."
            )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    result = None
    with col1:
        st.header("Upload Image")
        uploaded = st.file_uploader("Choose dermoscopy image", type=["jpg", "jpeg", "png"])
        
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded", use_container_width=True)
            
            temp_path = OUTPUT_DIR / "temp_upload.jpg"
            image.save(temp_path)
            
            with st.spinner("Analyzing..."):
                predictor = load_predictor()
                result = predictor.predict(temp_path)
    
    with col2:
        if result:
            st.header("Results")
            
            if not result['is_trained']:
                st.warning("⚠️ Model not trained yet! Predictions are RANDOM.")
                st.info("To train: Run `python src/train.py` in terminal, then refresh this app.")
            
            if result['is_malignant']:
                st.error(f"⚠️ Potential Malignancy: {result['prediction']}")
            else:
                st.success(f"✅ Benign: {result['prediction']}")
            
            st.metric("Confidence", f"{result['confidence']}%")
            
            st.subheader("Probabilities")
            for cls_name, prob in result['all_probabilities'].items():
                st.progress(prob / 100, text=f"{cls_name}: {prob}%")
    
    st.markdown("---")
    st.warning("⚠️ For educational purposes only. Consult a dermatologist.")


if __name__ == "__main__":
    main()
