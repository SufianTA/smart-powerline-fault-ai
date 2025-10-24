import streamlit as st
import os, torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from src.inference.infer import load_model
from src.training.utils import device

st.set_page_config(page_title="Power Line Fault Detector", page_icon="⚡", layout="centered")

st.title("⚡ Smart Fault Detection (Optical Sensor + Camera)")
st.write("Upload a signal `.npy` and a line image `.png/.jpg` to run inference with a multimodal model.")

ckpt = st.text_input("Checkpoint Path", value="runs/best_model.pt")

signal_file = st.file_uploader("Optical Sensor Signal (.npy)", type=["npy"])
image_file = st.file_uploader("Line Camera Image (.png, .jpg)", type=["png", "jpg", "jpeg"])

if st.button("Run Inference"):
    if not ckpt or not os.path.exists(ckpt):
        st.error("Checkpoint not found. Train a model first, then provide the path (e.g., runs/best_model.pt).")
    elif not signal_file or not image_file:
        st.error("Please upload both a signal and an image.")
    else:
        # Load model
        model, cfg = load_model(ckpt)
        # Prepare inputs
        signal = np.load(signal_file).astype(np.float32)
        L = cfg['data']['signal_length']
        if len(signal) != L:
            if len(signal) > L:
                signal = signal[:L]
            else:
                signal = np.pad(signal, (0, L - len(signal)), mode='constant')
        signal_t = torch.from_numpy(signal)[None,None,:].to(device())

        img = Image.open(image_file).convert('L')
        tf = T.Compose([T.Resize((cfg['data']['image_size'], cfg['data']['image_size'])), T.ToTensor()])
        img_t = tf(img)[None,:].to(device())

        with torch.no_grad():
            logits = model(signal_t, img_t)
            prob_fault = torch.softmax(logits, dim=1)[0,1].item()
            pred = int(logits.argmax(dim=1).item())

        st.subheader("Result")
        st.write(f"**Prediction:** {'FAULT' if pred==1 else 'NORMAL'}")
        st.write(f"**Fault Probability:** {prob_fault:.3f}")
        st.image(img, caption="Input Image", use_column_width=True)
        st.line_chart(signal)
