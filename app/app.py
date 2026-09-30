import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
import sys
import os
import io
import zipfile

# ─── Page config ───
st.set_page_config(page_title="Brain Tumor MRI AI", page_icon="🧠", layout="centered")

# ─── Design (CSS) ───
st.markdown("""
<style>
    .block-container {max-width: 960px; padding: 2rem 2rem 1rem 2rem;}
    #MainMenu, footer {visibility: hidden;}

    .hero {background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
           padding: 34px 30px; border-radius: 16px; color: white; margin-bottom: 22px;}
    .hero h1 {margin: 0 0 8px 0; font-size: 2.1rem; color: white;}
    .hero p {margin: 0; font-size: 1.02rem; opacity: .93;}

    .card {background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
           padding: 18px; height: 100%; color: #0f172a;
           box-shadow: 0 1px 3px rgba(0,0,0,.06);}
    .card h4 {margin: 0 0 6px 0; color: #1e3a8a; font-size: 1.05rem;}
    .card p {margin: 0; color: #475569; font-size: .92rem;}

    .section-title {font-size: 1.35rem; font-weight: 700; margin: 26px 0 8px 0;}

    .result {border-radius: 14px; padding: 22px; color: white; margin: 14px 0 18px 0;}
    .result .tag {margin: 0; font-size: .8rem; letter-spacing: .08em; opacity: .9;}
    .result h2 {margin: 4px 0 0 0; color: white; font-size: 1.6rem;}
    .result p {margin: 8px 0 0 0; opacity: .95;}

    .bar-label {display: flex; justify-content: space-between; font-size: .92rem; margin-top: 12px;}
    .bar-bg {background: #e2e8f0; border-radius: 8px; height: 12px; overflow: hidden;}
    .bar-fill {height: 12px; border-radius: 8px;}

    .warn {background: #fef3c7; border-left: 5px solid #f59e0b; color: #78350f;
           padding: 14px 18px; border-radius: 8px; margin-top: 26px; font-size: .92rem;}
    .footer {text-align: center; color: #64748b; font-size: .86rem; padding: 16px 0 4px 0;}
</style>
""", unsafe_allow_html=True)

# ─── Directories ───
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))

from gradcam import GradCAM

# ─── Load model ───
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

# ─── Sample ZIP ───
sample_dir = os.path.join(ROOT_DIR, "samples")

@st.cache_data
def create_sample_zip():
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(sample_dir):
            for folder in ["glioma", "meningioma", "notumor", "pituitary"]:
                folder_path = os.path.join(sample_dir, folder)
                if os.path.exists(folder_path):
                    for filename in sorted(os.listdir(folder_path)):
                        filepath = os.path.join(folder_path, filename)
                        if os.path.isfile(filepath):
                            zf.write(filepath, f"{folder}/{filename}")
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# ─── Sidebar: developer + model info ───
photo_path = os.path.join(APP_DIR, "../app/professional_cv_headshot.png")
with st.sidebar:
    st.markdown("## 👤 Developer")
    if os.path.exists(photo_path):
        st.image(photo_path, width=200)
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
for col, (title, text) in zip((s1, s2, s3), [
    ("1️⃣ Upload", "Choose an MRI image (JPG or PNG)."),
    ("2️⃣ Analyze", "The AI classifies the scan in seconds."),
    ("3️⃣ Understand", "See the prediction, confidence and attention map."),
]):
    col.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

# ─── Upload ───
st.markdown('<div class="section-title">📤 Upload MRI Scan</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"],
                                 label_visibility="collapsed")

# ─── Samples (collapsed so the page stays clean) ───
with st.expander("📥 No MRI scan? Download sample Data images for testing"):
    st.write("The ZIP has 4 folders with 10 images each (40 total): "
             "`glioma`, `meningioma`, `notumor`, `pituitary`.")
    st.download_button(
        label="⬇️ Download All Samples Data (ZIP)",
        data=create_sample_zip(),
        file_name="brain_tumor_mri_samples.zip",
        mime="application/zip",
        use_container_width=True,
    )

# ─── Results ───
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # BLOCK non-grayscale images (screenshots, photos, diagrams)
    img_array = np.array(image)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    color_diff = np.mean(np.abs(r.astype(float) - g.astype(float))) + np.mean(np.abs(g.astype(float) - b.astype(float)))
    
    if color_diff > 5:
        st.error("❌ This does not appear to be a brain MRI scan. MRI images are grayscale.")
        st.info("💡 Please upload a real brain MRI image, or download samples above.")
        st.stop()  # STOP - no prediction, no heatmap
    
    if color_diff > 25:
        st.error("❌ This does not appear to be a brain MRI scan. MRI images are grayscale.")
        st.info("💡 Please upload a real brain MRI image, or download samples above.")
        st.stop()  # STOP - no prediction, no heatmap
    
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

    # Confidence warning
    if confidence < 75:
        st.warning(f"⚠️ Low confidence ({confidence:.1f}%). This scan may need review by a radiologist.")
    elif confidence < 85:
        st.info(f"ℹ️ Moderate confidence ({confidence:.1f}%). Consider confirmation.")

    st.markdown(f"""
    <div class="result" style="background:{CLASS_COLORS[pred_name]};">
        <p class="tag">AI PREDICTION</p>
        <h2>{pred_name} · {confidence:.1f}% confidence</h2>
        <p>{CLASS_INFO[pred_name]}</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.image(image, caption="Original MRI", use_container_width=True)
    c2.image(overlay, caption="AI Attention Map (red = most important area)",
             use_container_width=True)

    st.markdown('<div class="section-title">📊 Probability for each class</div>', unsafe_allow_html=True)
    for i, name in enumerate(CLASS_NAMES):
        p = float(probabilities[i]) * 100
        st.markdown(f"""
        <div class="bar-label"><span>{name}</span><b>{p:.1f}%</b></div>
        <div class="bar-bg"><div class="bar-fill" style="width:{p}%; background:{CLASS_COLORS[name]};"></div></div>
        """, unsafe_allow_html=True)

    with st.expander("ℹ️ How to read the attention map"):
        st.write("The heatmap (Grad-CAM) highlights the image regions that influenced the AI's "
                 "decision most. Red and yellow areas mattered most; blue areas mattered least.")

# ─── Disclaimer + footer ───
st.markdown("""
<div class="warn">
⚠️ <b>Research demo only.</b> This tool is not a medical device and must not be used
for diagnosis. Always consult a qualified doctor.
</div>
<div class="footer">© 2026 书佐尔 (Bary M Shozaul) · 徐州工程学院 · rabby16139@gmail.com</div>
""", unsafe_allow_html=True)