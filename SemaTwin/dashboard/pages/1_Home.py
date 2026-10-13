import streamlit as st
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.ui_components import render_dashboard_header

st.set_page_config(page_title="1. 🏠 Overview", page_icon="🏠", layout="wide")

render_dashboard_header(current_stage="Overview")

# --- TITLE & SYSTEM ONLINE STATUS BADGE ---
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; background: #0F172A; border: 1px solid #334155; border-radius: 12px; padding: 20px 24px;">
        <div>
            <h1 style="color: #F8FAFC; font-weight: 800; font-size: 28px; letter-spacing: -0.5px; margin: 0;">
                DIGITAL TWIN FOR SEMANTIC COMMUNICATION
            </h1>
            <p style="color: #38BDF8; font-size: 15px; font-weight: 500; margin-top: 6px; margin-bottom: 0;">
                "AI-powered semantic image transmission over a simulated 6G network"
            </p>
        </div>
        <div>
            <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; color: #10B981; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px; letter-spacing: 0.5px; display: inline-flex; align-items: center; gap: 8px;">
                <span style="height: 10px; width: 10px; background-color: #10B981; border-radius: 50%; display: inline-block; box-shadow: 0 0 10px #10B981;"></span>
                ● SYSTEM ONLINE
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- 11-STAGE VISUAL ARCHITECTURE PIPELINE ---
st.markdown("### 🔄 End-to-End System Pipeline Architecture")
pipeline_html = """
<div style="background: #0F172A; border: 1px solid #334155; border-radius: 12px; padding: 22px; margin-bottom: 24px; box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
    <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 8px;">
        <div style="background: #1E293B; border: 1px solid #0284C7; padding: 10px 14px; border-radius: 8px; color: #38BDF8; font-weight: 700; font-size: 12px; text-align: center; min-width: 95px;">
            🖼️ IMAGE
        </div>
        <div style="color: #64748B; font-weight: 800;">➔</div>
        <div style="background: #1E293B; border: 1px solid #2563EB; padding: 10px 14px; border-radius: 8px; color: #60A5FA; font-weight: 700; font-size: 12px; text-align: center; min-width: 100px;">
            🧠 RESNET-18
        </div>
        <div style="color: #64748B; font-weight: 800;">➔</div>
        <div style="background: #1E293B; border: 1px solid #7C3AED; padding: 10px 14px; border-radius: 8px; color: #A78BFA; font-weight: 700; font-size: 12px; text-align: center; min-width: 100px;">
            🔬 SEMANTIC
        </div>
        <div style="color: #64748B; font-weight: 800;">➔</div>
        <div style="background: #1E293B; border: 1px solid #059669; padding: 10px 14px; border-radius: 8px; color: #34D399; font-weight: 700; font-size: 12px; text-align: center; min-width: 95px;">
            🖥️ SERVER
        </div>
        <div style="color: #64748B; font-weight: 800;">➔</div>
        <div style="background: #1E293B; border: 1px solid #10B981; padding: 10px 14px; border-radius: 8px; color: #6EE7B7; font-weight: 700; font-size: 12px; text-align: center; min-width: 90px;">
            📱 NODE
        </div>
        <div style="color: #64748B; font-weight: 800;">➔</div>
        <div style="background: #1E293B; border: 1px solid #D97706; padding: 10px 14px; border-radius: 8px; color: #FBBF24; font-weight: 700; font-size: 12px; text-align: center; min-width: 100px;">
            📡 AWGN / RAYLEIGH
        </div>
    </div>
    <div style="text-align: center; margin: 10px 0; color: #64748B; font-weight: 800;">↓</div>
    <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 8px;">
        <div style="background: #1E293B; border: 1px solid #DB2777; padding: 10px 14px; border-radius: 8px; color: #F472B6; font-weight: 700; font-size: 12px; text-align: center; min-width: 110px;">
            🪞 DIGITAL TWIN
        </div>
        <div style="color: #64748B; font-weight: 800;">➔</div>
        <div style="background: #1E293B; border: 1px solid #9333EA; padding: 10px 14px; border-radius: 8px; color: #C084FC; font-weight: 700; font-size: 12px; text-align: center; min-width: 90px;">
            🔮 LSTM
        </div>
        <div style="color: #64748B; font-weight: 800;">➔</div>
        <div style="background: #1E293B; border: 1px solid #EA580C; padding: 10px 14px; border-radius: 8px; color: #FB923C; font-weight: 700; font-size: 12px; text-align: center; min-width: 130px;">
            🎛️ ADAPTIVE CONTROL
        </div>
        <div style="color: #64748B; font-weight: 800;">➔</div>
        <div style="background: #1E293B; border: 1px solid #2563EB; padding: 10px 14px; border-radius: 8px; color: #60A5FA; font-weight: 700; font-size: 12px; text-align: center; min-width: 110px;">
            ⚡ DEEP JSCC
        </div>
        <div style="color: #64748B; font-weight: 800;">➔</div>
        <div style="background: #1E293B; border: 1px solid #10B981; padding: 10px 14px; border-radius: 8px; color: #34D399; font-weight: 700; font-size: 12px; text-align: center; min-width: 130px;">
            🖼️ RECONSTRUCTION
        </div>
    </div>
</div>
"""
st.markdown(pipeline_html, unsafe_allow_html=True)

# --- SIX SYSTEM SPECIFICATION METRICS ---
st.markdown("### 📊 System Key Specifications & Infrastructure")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric(label="Dataset Benchmark", value="CIFAR-10", delta="10 Classes")
with c2:
    st.metric(label="Semantic Encoder", value="RESNET-18", delta="Pretrained CNN")
with c3:
    st.metric(label="Feature Vector", value="512-D FEATURES", delta="Latent Space")
with c4:
    st.metric(label="Edge Servers", value="2 SERVERS", delta="Server 1 & 2")
with c5:
    st.metric(label="Edge Nodes", value="10 NODES", delta="N1 through N10")
with c6:
    st.metric(label="Channel Codec", value="DEEP JSCC", delta="2 Channel Models: AWGN + Rayleigh Fading")

st.markdown("---")

# --- EXPERIMENT START PROMPT ---
st.markdown("### 🚀 Start Experiment Pipeline")
st.caption("Begin the continuous experiment sequence: Image & Semantics ➔ Network ➔ Communication ➔ Digital Twin ➔ Prediction ➔ Control ➔ Results.")

if st.button("🚀 Start Experiment ➔ Go to Image & Semantics", key="btn_home_start_exp", use_container_width=True, type="primary"):
    try:
        st.switch_page("pages/2_Image_and_Semantics.py")
    except Exception:
        st.info("👈 Select **🖼️ Image & Semantics** in the sidebar navigation to begin.")

