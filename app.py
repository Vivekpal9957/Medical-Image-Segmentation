"""
Medical Image Segmentation - Streamlit Web App
Run: streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import os

from model import UNet
from utils import dice_coefficient, iou_score


# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Medical Segmentation | U-Net",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #e94560;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 16px;
        color: #888;
        text-align: center;
        margin-bottom: 25px;
    }
    .metric-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #e94560;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #e94560;
    }
    .metric-label {
        font-size: 12px;
        color: #aaa;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=1).to(device)
    model_path = 'checkpoints/best_model.pth'
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # IMPORTANT FIX: Handle checkpoint dict format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model, device, True
    else:
        return model, device, False


# ============================================
# PREPROCESS IMAGE
# ============================================
def preprocess_image(image):
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((256, 256))
    img_array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    return tensor, img_array


# ============================================
# PREDICT
# ============================================
def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        prob = torch.sigmoid(output)
        binary = (prob > 0.5).float()
        return prob.cpu().numpy().squeeze(), binary.cpu().numpy().squeeze()


# ============================================
# VISUALIZATION
# ============================================
def create_result_figure(original, prob_mask, binary_mask):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('#0e1117')
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Scan', fontsize=13, fontweight='bold', color='white')
    axes[0].axis('off')
    
    im = axes[1].imshow(prob_mask, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Tumor Probability', fontsize=13, fontweight='bold', color='white')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    overlay = np.zeros((*original.shape, 3))
    overlay[..., 0] = original
    overlay[..., 1] = original
    overlay[..., 2] = original
    overlay[..., 0] = np.maximum(overlay[..., 0], binary_mask * 0.85)
    overlay[..., 1] = np.maximum(overlay[..., 1], binary_mask * 0.15)
    overlay[..., 2] = np.maximum(overlay[..., 2], binary_mask * 0.15)
    
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title('Segmentation Overlay', fontsize=13, fontweight='bold', color='white')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


# ============================================
# MAIN APP
# ============================================
def main():
    st.markdown('<p class="main-title">🏥 Medical Image Segmentation</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Tumor Detection using U-Net Deep Learning</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("⚙️ Controls")
        st.markdown("---")
        
        model, device, model_loaded = load_model()
        
        if model_loaded:
            st.success("✅ Model loaded!")
            st.info(f"Device: **{str(device).upper()}**")
        else:
            st.error("❌ Model not found!")
            st.warning("Train first: `python main.py`")
        
        st.markdown("---")
        st.markdown("""
        **How to use:**
        1. Upload a medical scan
        2. Click **Analyze Scan**
        3. View results
        4. Download mask
        """)
        
        st.markdown("---")
        st.markdown("**Model:** U-Net | **Input:** 1×256×256 | **Best Dice:** 0.92")
    
    if not model_loaded:
        st.error("🚫 No trained model found at `checkpoints/best_model.pth`")
        st.info("👉 Run `python main.py` first to train the model.")
        return
    
    uploaded_file = st.file_uploader(
        "📤 Upload Medical Scan",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a grayscale medical image (CT, MRI, X-ray)"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, use_container_width=True)
        
        tensor, original_array = preprocess_image(image)
        
        with col2:
            st.subheader("🔬 Analysis")
            if st.button("Analyze Scan", type="primary", use_container_width=True):
                with st.spinner("AI is analyzing... Please wait ⏳"):
                    prob_mask, binary_mask = predict(model, tensor, device)
                
                st.success("✅ Analysis Complete!")
                
                tumor_pixels = int(np.sum(binary_mask))
                tumor_pct = (tumor_pixels / (256*256)) * 100
                max_conf = float(np.max(prob_mask))
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{tumor_pixels}</div><div class="metric-label">Tumor Pixels</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{tumor_pct:.2f}%</div><div class="metric-label">Coverage</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{max_conf:.3f}</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
                
                st.markdown("---")
                st.subheader("📊 Segmentation Results")
                
                fig = create_result_figure(original_array, prob_mask, binary_mask)
                st.pyplot(fig)
                
                st.markdown("---")
                st.subheader("📥 Download Result")
                
                mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
                buf = BytesIO()
                mask_img.save(buf, format='PNG')
                
                st.download_button(
                    label="⬇️ Download Segmentation Mask",
                    data=buf.getvalue(),
                    file_name="segmentation_mask.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                with st.expander("🔍 Compare with Ground Truth (Optional)"):
                    gt_file = st.file_uploader("Upload ground truth mask", type=['png', 'jpg', 'jpeg'])
                    if gt_file is not None:
                        gt_image = Image.open(gt_file).convert('L').resize((256, 256))
                        gt_array = np.array(gt_image).astype(np.float32) / 255.0
                        gt_tensor = torch.from_numpy(gt_array).unsqueeze(0).unsqueeze(0)
                        pred_tensor = torch.from_numpy(binary_mask).unsqueeze(0).unsqueeze(0)
                        
                        dice = dice_coefficient(pred_tensor, gt_tensor).item()
                        iou = iou_score(pred_tensor, gt_tensor).item()
                        
                        c1, c2 = st.columns(2)
                        c1.metric("Dice Score", f"{dice:.4f}")
                        c2.metric("IoU Score", f"{iou:.4f}")
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 12px;'>"
        "Built with Streamlit | Medical Image Segmentation Project | U-Net</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
