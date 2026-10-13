import streamlit as st
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib
import digital_twin.state
import digital_twin.twin
import communication.transmission
importlib.reload(digital_twin.state)
importlib.reload(digital_twin.twin)
importlib.reload(communication.transmission)

from data.dataset import CIFAR10DataModule
from models.resnet_semantic import ResNet18CIFAR10
from semantic.feature_extractor import ResNet18SemanticFeatureExtractor
from digital_twin.twin import DigitalTwinEngine
from communication.transmission import SemanticTransmissionPipeline
from dashboard.ui_components import render_dashboard_header

st.set_page_config(page_title="8. 📊 Results", page_icon="📊", layout="wide")

render_dashboard_header(current_stage="Results")

@st.cache_resource
def get_comm_models():
    dm = CIFAR10DataModule()
    m = ResNet18CIFAR10(num_classes=10, pretrained=True)
    pth_path = os.path.join(PROJECT_ROOT, "results", "models", "resnet18_cifar10.pth")
    if os.path.exists(pth_path):
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            m.load_state_dict(torch.load(pth_path, map_location=device))
        except Exception as e:
            print(f"[RESNET MODEL LOAD ERROR] {e}")
    m.eval()
    return dm, m

dm, model = get_comm_models()

def ensure_shared_experiment_image():
    if "selected_image" not in st.session_state or st.session_state["selected_image"] is None:
        try:
            extractor = ResNet18SemanticFeatureExtractor()
            pil_img, cls_name, _, _, img_path = dm.get_random_sample_by_class("dog")
            pred_res = model.predict_image(pil_img)
            ext_res = extractor.extract_semantic_features(pil_img)
            st.session_state["selected_image"] = pil_img
            st.session_state["actual_class"] = cls_name
            st.session_state["predicted_class"] = pred_res["predicted_class"].upper()
            st.session_state["confidence"] = pred_res["confidence_pct"]
            st.session_state["semantic_meaning"] = pred_res["predicted_class"].upper()
            st.session_state["semantic_features"] = ext_res["feature_vector"]
            st.session_state["image_path"] = img_path
        except Exception as e:
            print(f"[EXPERIMENT IMG INIT WARNING] {e}")

ensure_shared_experiment_image()

if "dt_engine" not in st.session_state:
    st.session_state["dt_engine"] = DigitalTwinEngine()

dt_engine = st.session_state["dt_engine"]
pipeline = SemanticTransmissionPipeline()

curr_img = st.session_state["selected_image"]
comm_class = st.session_state.get("actual_class", "dog").upper()
sem_meaning = st.session_state.get("semantic_meaning", comm_class).upper()
selected_node = st.session_state.get("selected_node", "N3")

source_server = pipeline.determine_source_server(selected_node)
target_telemetry = dt_engine.state.node_telemetry.get(selected_node, {
    "snr_db": 20.0, "latency_ms": 10.0, "bandwidth_mbps": 100.0, "packet_loss_pct": 0.1, "status": "ACTIVE"
})

# Retrieve existing transmission result or run live transmission
if "last_jscc_result" in st.session_state:
    res_j = st.session_state["last_jscc_result"]
elif "page5_last_jscc_result" in st.session_state:
    res_j = st.session_state["page5_last_jscc_result"]
else:
    res_j = pipeline.transmit_real_jscc(
        pil_or_np_img=curr_img,
        destination_node_id=selected_node,
        node_telemetry=target_telemetry,
        dt_engine=dt_engine
    )

st.markdown("### 📊 Experiment Results & Semantic Transmission Evaluation")
st.caption("End-to-End Deep JSCC Transmission and Digital Twin Telemetry Metrics")

# --- 1. ORIGINAL IMAGE vs RECONSTRUCTED IMAGE (STREAMLIT COLUMNS) ---
st.markdown("---")
st.markdown("#### 🖼️ IMAGE COMPARISON")

col_img1, col_img2 = st.columns(2)

with col_img1:
    st.markdown("**ORIGINAL IMAGE**")
    st.image(res_j["original_np"], caption=f"Original Input Image ({comm_class})", use_container_width=True)

with col_img2:
    st.markdown("**RECONSTRUCTED IMAGE**")
    st.image(res_j["reconstructed_np"], caption=f"Reconstructed Image ({selected_node})", use_container_width=True)

st.markdown("---")

# --- 2. SIX PERFORMANCE METRICS ---
st.markdown("#### 📈 PERFORMANCE METRICS")

col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
col_m1.metric("MSE", f"{res_j.get('mse', 0.0):.6f}")
col_m2.metric("PSNR", f"{res_j.get('psnr_db', 0.0):.2f} dB")
col_m3.metric("SNR", f"{res_j.get('snr_db', 0.0):.2f} dB")
col_m4.metric("PACKET LOSS", f"{res_j.get('packet_loss_pct', 0.0):.2f}%")
col_m5.metric("LATENCY", f"{res_j.get('latency_ms', 0.0):.2f} ms")
col_m6.metric("BANDWIDTH", f"{res_j.get('bandwidth_mbps', 0.0):.2f} Mbps")

st.markdown("---")

# --- 3. STATUS & METADATA CARDS ---
st.markdown("#### 📋 TRANSMISSION & RECONSTRUCTION METADATA")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    st.markdown(
        f"""
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 10px; padding: 18px; text-align: center;">
            <span style="color: #94A3B8; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;">SEMANTIC</span>
            <div style="color: #38BDF8; font-size: 22px; font-weight: 800; margin-top: 4px;">{sem_meaning}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_s2:
    st.markdown(
        f"""
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 10px; padding: 18px; text-align: center;">
            <span style="color: #94A3B8; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;">DESTINATION</span>
            <div style="color: #F59E0B; font-size: 22px; font-weight: 800; margin-top: 4px;">{selected_node}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_s3:
    st.markdown(
        """
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 10px; padding: 18px; text-align: center;">
            <span style="color: #94A3B8; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;">TRANSMISSION</span>
            <div style="color: #10B981; font-size: 20px; font-weight: 800; margin-top: 4px;">COMPLETED</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_s4:
    st.markdown(
        """
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 10px; padding: 18px; text-align: center;">
            <span style="color: #94A3B8; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;">RECONSTRUCTION</span>
            <div style="color: #10B981; font-size: 20px; font-weight: 800; margin-top: 4px;">COMPLETED</div>
        </div>
        """,
        unsafe_allow_html=True
    )

import plotly.graph_objects as go
import pandas as pd

@st.cache_data(show_spinner=False)
def compute_snr_vs_psnr_sweep(img_np, seed=42):
    """
    Computes empirical SNR vs. PSNR performance sweep for AWGN vs Rayleigh Fading
    across SNR values from 0 dB to 20 dB (21 SNR points x 2 channels = 42 measurements).
    Uses a fixed random seed for 100% reproducible evaluation.
    """
    from models.deep_jscc import FastDeepJSCCEngine
    engine = FastDeepJSCCEngine(latent_channels=16)

    img_np = np.clip(np.array(img_np, dtype=np.float32), 0.0, 1.0)
    if img_np.max() > 1.0:
        img_np /= 255.0

    snr_range = list(range(0, 21))
    awgn_psnr_list = []
    rayleigh_psnr_list = []
    awgn_mse_list = []
    rayleigh_mse_list = []

    np.random.seed(seed)

    for snr in snr_range:
        res_a = engine.process_image(img_np, snr_db=float(snr), channel_type="AWGN")
        res_r = engine.process_image(img_np, snr_db=float(snr), channel_type="Rayleigh")
        
        awgn_psnr_list.append(round(float(res_a["psnr_db"]), 2))
        rayleigh_psnr_list.append(round(float(res_r["psnr_db"]), 2))
        awgn_mse_list.append(round(float(res_a["mse"]), 6))
        rayleigh_mse_list.append(round(float(res_r["mse"]), 6))

    return {
        "snr_range": snr_range,
        "awgn_psnr": awgn_psnr_list,
        "rayleigh_psnr": rayleigh_psnr_list,
        "awgn_mse": awgn_mse_list,
        "rayleigh_mse": rayleigh_mse_list
    }

# --- NEW SECTION: SNR VS. PSNR PERFORMANCE EVALUATION SWEEP ---
st.markdown("---")
st.markdown("#### 📈 6G CHANNEL EVALUATION: SNR vs. PSNR SWEEP (0 to 20 dB)")
st.caption("Empirical performance evaluation across SNR values from 0 dB to 20 dB (21 SNR points × 2 Channel Models = 42 actual measurements) for input image category `" + comm_class + "`.")

sweep_data = compute_snr_vs_psnr_sweep(curr_img)

fig_sweep = go.Figure()

fig_sweep.add_trace(go.Scatter(
    x=sweep_data["snr_range"],
    y=sweep_data["awgn_psnr"],
    mode="lines+markers",
    name="AWGN Channel",
    line=dict(color="#38BDF8", width=3),
    marker=dict(size=7, color="#0EA5E9", symbol="circle")
))

fig_sweep.add_trace(go.Scatter(
    x=sweep_data["snr_range"],
    y=sweep_data["rayleigh_psnr"],
    mode="lines+markers",
    name="Rayleigh Fading Channel",
    line=dict(color="#F59E0B", width=3, dash="dash"),
    marker=dict(size=7, color="#D97706", symbol="diamond")
))

fig_sweep.update_layout(
    template="plotly_dark",
    title="Deep JSCC Image Quality Evaluation: AWGN vs. Rayleigh Fading (0 to 20 dB)",
    height=420,
    margin=dict(l=20, r=20, t=50, b=20),
    paper_bgcolor="#0F172A",
    plot_bgcolor="#0F172A",
    xaxis=dict(title="Signal-to-Noise Ratio (SNR in dB)", dtick=1, range=[-0.5, 20.5], showgrid=True, gridcolor="#1E293B"),
    yaxis=dict(title="Peak Signal-to-Noise Ratio (PSNR in dB)", showgrid=True, gridcolor="#1E293B"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_sweep, use_container_width=True)

with st.expander("📋 View Complete Measured 42-Point SNR vs. PSNR Evaluation Data Table", expanded=False):
    df_sweep = pd.DataFrame({
        "SNR (dB)": [f"{s} dB" for s in sweep_data["snr_range"]],
        "AWGN MSE": sweep_data["awgn_mse"],
        "AWGN PSNR (dB)": [f"{p:.2f} dB" for p in sweep_data["awgn_psnr"]],
        "Rayleigh MSE": sweep_data["rayleigh_mse"],
        "Rayleigh PSNR (dB)": [f"{p:.2f} dB" for p in sweep_data["rayleigh_psnr"]],
        "PSNR Delta (dB)": [f"+{(a - r):.2f} dB" for a, r in zip(sweep_data["awgn_psnr"], sweep_data["rayleigh_psnr"])]
    })
    st.dataframe(df_sweep, use_container_width=True, hide_index=True)

st.markdown("---")

# --- RESTART EXPERIMENT BUTTON ---
if st.button("🔄 Restart Experiment Pipeline ➔ Overview", key="btn_restart_exp_p8", use_container_width=True, type="primary"):
    try:
        st.switch_page("pages/1_Home.py")
    except Exception:
        st.info("👈 Select **🏠 Overview** in the sidebar navigation to restart.")
