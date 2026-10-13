import streamlit as st
import numpy as np
import os
import sys
import plotly.express as px
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib
import data.dataset
importlib.reload(data.dataset)
from data.dataset import CIFAR10DataModule, CIFAR10_CLASSES
from models.resnet_semantic import ResNet18CIFAR10
from semantic.feature_extractor import ResNet18SemanticFeatureExtractor
from dashboard.ui_components import render_dashboard_header

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

st.set_page_config(page_title="2. 🖼️ Image & Semantics", page_icon="🖼️", layout="wide")

render_dashboard_header(current_stage="Image & Semantics")

def get_data_module():
    return data.dataset.CIFAR10DataModule()

@st.cache_resource
def get_model():
    m = ResNet18CIFAR10(num_classes=10, pretrained=True)
    pth_path = os.path.join(PROJECT_ROOT, "results", "models", "resnet18_cifar10.pth")
    if TORCH_OK and os.path.exists(pth_path) and hasattr(torch, "device"):
        try:
            device = torch.device("cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu")
            m.load_state_dict(torch.load(pth_path, map_location=device))
        except Exception as e:
            print(f"[MODEL LOAD WARNING] {e}")
    if hasattr(m, "eval"):
        m.eval()
    return m

@st.cache_resource
def get_feature_extractor():
    return ResNet18SemanticFeatureExtractor()

dm = get_data_module()
model = get_model()
extractor = get_feature_extractor()

def update_shared_experiment_image(pil_img, cls_name, img_path):
    pred_res = model.predict_image(pil_img)
    ext_res = extractor.extract_semantic_features(pil_img)

    st.session_state["selected_image"] = pil_img
    st.session_state["actual_class"] = cls_name
    st.session_state["predicted_class"] = pred_res["predicted_class"].upper()
    st.session_state["confidence"] = pred_res["confidence_pct"]
    st.session_state["semantic_meaning"] = pred_res["predicted_class"].upper()
    st.session_state["semantic_features"] = ext_res["feature_vector"]
    st.session_state["image_path"] = img_path

    st.session_state["pipeline_img"] = pil_img
    st.session_state["pipeline_class"] = cls_name
    st.session_state["pipeline_path"] = img_path

if "selected_image" not in st.session_state or st.session_state["selected_image"] is None:
    try:
        pil_img, cls_name, _, _, img_path = dm.get_correctly_classified_sample_by_class("dog", model=model)
        update_shared_experiment_image(pil_img, cls_name, img_path)
    except Exception as e:
        print(f"[EXPERIMENT IMG INIT WARNING] {e}")

# --- MAIN INTERACTIVE LAYOUT (LEFT: IMAGE | RIGHT: METRICS & 512-D FEATURES) ---
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.markdown("### 🖼️ Selected CIFAR-10 Image")
    selected_class = st.selectbox(
        "Select CIFAR-10 Class Category:",
        CIFAR10_CLASSES,
        index=CIFAR10_CLASSES.index(st.session_state.get("actual_class", "dog").lower()) if st.session_state.get("actual_class", "dog").lower() in CIFAR10_CLASSES else 5,
        key="p2_exp_cls_select"
    )
    if "last_selected_exp_class_p2" not in st.session_state or st.session_state.get("last_selected_exp_class_p2") != selected_class:
        pil_img, cls_name, cls_idx, img_np, img_path = dm.get_correctly_classified_sample_by_class(selected_class, model=model)
        update_shared_experiment_image(pil_img, cls_name, img_path)
        st.session_state["last_selected_exp_class_p2"] = selected_class

    if st.button("🎲 Fetch Another Sample Image", key="p2_exp_fetch_btn", use_container_width=True):
        pil_img, cls_name, cls_idx, img_np, img_path = dm.get_correctly_classified_sample_by_class(selected_class, model=model)
        update_shared_experiment_image(pil_img, cls_name, img_path)

    st.image(st.session_state["selected_image"], caption=f"Selected Image ({st.session_state['actual_class'].upper()})", use_container_width=True)

with col_right:
    st.markdown("### 🧠 Semantic Analysis")
    
    act_cls = st.session_state['actual_class'].upper()
    pred_cls = st.session_state['predicted_class'].upper()
    conf_pct = st.session_state['confidence']
    sem_meaning = st.session_state['semantic_meaning']
    
    st.markdown(
        f"""
        <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 10px; padding: 18px; margin-bottom: 16px;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; font-size: 13px;">
                <div><span style="color:#94A3B8;">Actual Class:</span><br><b style="color:#F8FAFC; font-size: 15px;">{act_cls}</b></div>
                <div><span style="color:#94A3B8;">Predicted Class:</span><br><b style="color:#38BDF8; font-size: 15px;">{pred_cls}</b></div>
                <div><span style="color:#94A3B8;">Confidence:</span><br><b style="color:#10B981; font-size: 15px;">{conf_pct:.2f}%</b></div>
                <div><span style="color:#94A3B8;">Semantic Meaning:</span><br><b style="color:#F59E0B; font-size: 15px;">{sem_meaning}</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("#### 🔬 512-D Feature Representation")
    feat_vec = np.array(st.session_state["semantic_features"])
    
    fig_feat = px.bar(
        x=list(range(len(feat_vec))),
        y=feat_vec,
        labels={"x": "Feature Index (0..511)", "y": "Activation"},
        title=f"512-D Feature Activations for `{act_cls}`"
    )
    fig_feat.update_layout(
        template="plotly_dark",
        height=240,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A"
    )
    st.plotly_chart(fig_feat, use_container_width=True)

st.markdown("---")

if st.button("Continue → Network Topology", key="btn_continue_to_net_p2", use_container_width=True, type="primary"):
    try:
        st.switch_page("pages/3_Network_Topology.py")
    except Exception:
        st.info("👈 Select **🌐 Network** in the sidebar navigation to proceed.")
