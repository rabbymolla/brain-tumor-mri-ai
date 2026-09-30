import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
import sys
import os

# Get directories
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))

from gradcam import GradCAM

# ─── Load Model ───
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    
    model_path = os.path.join(ROOT_DIR, "results", "brain_tumor_resnet18.pth")
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    gradcam = GradCAM(model, model.layer4[1].conv2)
    return model, gradcam, device

model, gradcam, device = load_model()

CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ─── Navbar ───
st.markdown("""
<div style="background-color: #1e3a8a; padding: 15px; border-radius: 10px;">
    <h2 style="color: white; margin: 0;">🧠 Brain Tumor MRI AI</h2>
</div>
""", unsafe_allow_html=True)

# ─── Your Info ───
photo_path = os.path.join(APP_DIR, "photo.jpg")
col1, col2 = st.columns([1, 3])
with col1:
    if os.path.exists(photo_path):
        st.image(photo_path, width=150)
    else:
        st.write("📷")
with col2:
    st.markdown("""
    ### Bary M Shozaul (书佐尔)
    **Xuzhou University of Technology (徐州工程学院)**  
    Computer Science & Technology  
    📧 rabby16139@gmail.com
    """)

st.divider()

# ─── Upload ───
st.header("Upload MRI Scan")
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    # Grad-CAM
    cam, predicted = gradcam.generate(input_tensor)
    cam = cv2.resize(cam, (224, 224))
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_array = np.array(image.resize((224, 224)))
    overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
    
    # Show results
    c1, c2 = st.columns(2)
    with c1:
        st.image(image, caption="Original MRI", use_container_width=True)
    with c2:
        st.image(overlay, caption="AI Attention Map", use_container_width=True)
    
    # Prediction text
    pred_name = CLASS_NAMES[predicted]
    confidence = float(probabilities[predicted]) * 100
    
    st.success(f"Prediction: **{pred_name}** ({confidence:.2f}% confidence)")
    
    # Probability bars
    st.subheader("All Classes:")
    for i, name in enumerate(CLASS_NAMES):
        prob = float(probabilities[i]) * 100
        st.progress(prob / 100, text=f"{name}: {prob:.1f}%")

# ─── Footer ───
st.divider()
st.markdown("""
<div style="background-color: #0f172a; color: white; padding: 20px; border-radius: 10px; text-align: center;">
    <p><strong>© 2026 书佐尔 (Bary M Shozaul) | 徐州工程学院</strong></p>
    <p>Contact: rabby16139@gmail.com</p>
    <p style="font-size: 12px;">⚠️ For academic research only. Not for clinical diagnosis.</p>
</div>
""", unsafe_allow_html=True)