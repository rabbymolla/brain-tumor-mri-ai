import streamlit as st
import torch
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import timm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from gradcam import GradCAM

# ─── Page setup ───
st.set_page_config(page_title="Brain Tumor MRI AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    .hero {background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
           padding: 36px 32px; border-radius: 16px; color: white; margin-bottom: 24px;}
    .hero h1 {margin: 0 0 8px 0; font-size: 2.2rem; color: white;}
    .hero p {margin: 0; font-size: 1.05rem; opacity: .92;}
    .card {background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
           padding: 20px; height: 100%; color: #0f172a;
           box-shadow: 0 1px 3px rgba(0,0,0,.06);}
    .card h4 {margin: 0 0 6px 0; color: #1e3a8a;}
    .card p {margin: 0; color: #475569; font-size: .95rem;}
    .result {border-radius: 14px; padding: 22px; color: white; margin: 12px 0 20px 0;}
    .result h2 {margin: 0; color: white;}
    .result p {margin: 6px 0 0 0; opacity: .95;}
    .bar-label {display:flex; justify-content:space-between; font-size:.92rem; margin-top:10px;}
    .bar-bg {background:#e2e8f0; border-radius:8px; height:12px; overflow:hidden;}
    .bar-fill {height:12px; border-radius:8px;}
    .warn {background:#fef3c7; border-left: 5px solid #f59e0b; color:#78350f;
           padding: 14px 18px; border-radius: 8px; margin-top: 20px; font-size:.92rem;}
    .footer {text-align:center; color:#64748b; font-size:.88rem; padding: 18px 0 6px 0;}
    #MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Load model ───
@st.cache_resource
def load_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = timm.create_model("resnet18", pretrained=False, num_classes=4)
    model.load_state_dict(torch.load("../results/brain_tumor_resnet18.pth", map_location=device))
    model = model.to(device)
    model.eval()
    gradcam = GradCAM(model, model.layer4[1].conv2)
    return model, gradcam, device

model, gradcam, device = load_model()

CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]
CLASS_INFO = {
    "Glioma Tumor": "A tumor that starts in glial cells, the supporting cells of the brain.",
    "Meningioma Tumor": "A tumor that grows in the meninges, the layers covering the brain.",
    "No Tumor": "No tumor pattern was detected in this image.",
    "Pituitary Tumor": "A tumor in the pituitary gland, a small gland at the base of the brain.",
}
CLASS_COLORS = {
    "Glioma Tumor": "#dc2626",
    "Meningioma Tumor": "#ea580c",
    "No Tumor": "#16a34a",
    "Pituitary Tumor": "#7c3aed",
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# ─── Sidebar ───
with st.sidebar:
    st.markdown("## 👤 Developer")
    try:
        st.image("photo.jpg", width=140)
    except Exception:
        pass
    st.markdown("""
**Bary M Shozaul (书佐尔)**  
Computer Science & Technology  
Xuzhou University of Technology (徐州工程学院)

📧 rabby16139@gmail.com
""")
    st.divider()
    st.markdown("## ⚙️ Model")
    st.markdown("""
- **Network:** ResNet-18  
- **Classes:** 4  
- **Input size:** 224 × 224  
- **Explainability:** Grad-CAM
""")

# ─── Hero ───
st.markdown("""
<div class="hero">
    <h1>🧠 Brain Tumor MRI AI</h1>
    <p>Upload a brain MRI scan. The AI predicts the tumor type and shows
    <b>where it looked</b> in the image.</p>
</div>
""", unsafe_allow_html=True)

# ─── How it works ───
s1, s2, s3 = st.columns(3)
steps = [
    ("1️⃣ Upload", "Choose an MRI image (JPG or PNG)."),
    ("2️⃣ Analyze", "The AI classifies the scan in seconds."),
    ("3️⃣ Understand", "See the prediction, confidence, and attention map."),
]
for col, (title, text) in zip((s1, s2, s3), steps):
    col.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

st.write("")
st.markdown("### 📤 Upload MRI Scan")
uploaded_file = st.file_uploader("Drag and drop or browse", type=["jpg", "jpeg", "png"],
                                 label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analyzing scan..."):
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(input_tensor), dim=1)[0]

        cam, predicted = gradcam.generate(input_tensor)
        cam = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        img_array = np.array(image.resize((224, 224)))
        overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)

    pred_name = CLASS_NAMES[predicted]
    confidence = float(probabilities[predicted]) * 100
    color = CLASS_COLORS[pred_name]

    # Result banner
    st.markdown(f"""
    <div class="result" style="background:{color};">
        <p style="margin:0; font-size:.9rem;">AI PREDICTION</p>
        <h2>{pred_name} · {confidence:.1f}% confidence</h2>
        <p>{CLASS_INFO[pred_name]}</p>
    </div>
    """, unsafe_allow_html=True)

    # Images
    c1, c2 = st.columns(2)
    c1.image(image, caption="Original MRI", use_container_width=True)
    c2.image(overlay, caption="AI Attention Map (red = most important area)",
             use_container_width=True)

    # Probability bars
    st.markdown("### 📊 Probability for each class")
    for i, name in enumerate(CLASS_NAMES):
        p = float(probabilities[i]) * 100
        st.markdown(f"""
        <div class="bar-label"><span>{name}</span><b>{p:.1f}%</b></div>
        <div class="bar-bg"><div class="bar-fill" style="width:{p}%; background:{CLASS_COLORS[name]};"></div></div>
        """, unsafe_allow_html=True)

    with st.expander("ℹ️ How to read the attention map"):
        st.write("The colored heatmap (Grad-CAM) highlights the image regions that influenced "
                 "the AI's decision most. Red and yellow areas mattered most; blue areas mattered least.")

# ─── Disclaimer + footer ───
st.markdown("""
<div class="warn">
⚠️ <b>Research demo only.</b> This tool is not a medical device and must not be used
for diagnosis. Always consult a qualified doctor.
</div>
<div class="footer">© 2026 书佐尔 (Bary M Shozaul) · 徐州工程学院 · rabby16139@gmail.com</div>
""", unsafe_allow_html=True)