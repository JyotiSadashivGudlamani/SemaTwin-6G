import streamlit as st
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib
import digital_twin.state
import digital_twin.twin
import digital_twin.predictor
import control.adaptive_controller
importlib.reload(digital_twin.state)
importlib.reload(digital_twin.twin)
importlib.reload(digital_twin.predictor)
importlib.reload(control.adaptive_controller)

from digital_twin.twin import DigitalTwinEngine
from digital_twin.predictor import DigitalTwinPredictor
from control.adaptive_controller import ProactiveAdaptiveController
from dashboard.ui_components import render_dashboard_header

st.set_page_config(page_title="7. 🎛️ Adaptive Control", page_icon="🎛️", layout="wide")

render_dashboard_header(current_stage="Adaptive Control")

if "dt_engine" not in st.session_state:
    st.session_state["dt_engine"] = DigitalTwinEngine()

dt_engine = st.session_state["dt_engine"]
predictor = DigitalTwinPredictor()
controller = ProactiveAdaptiveController()

st.markdown("### 🎛️ Proactive 6G Adaptive Control Pipeline")
st.caption("Closed-loop semantic communication profile adaptation based on Digital Twin PyTorch LSTM channel forecasting")

# --- 1. VISUAL FLOW PIPELINE ARCHITECTURE ---
st.markdown("#### 🔄 Closed-Loop Control Flow Architecture")
st.markdown(
    """
    <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 12px; padding: 20px; margin-bottom: 24px;">
        <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 10px; text-align: center;">
            <div style="background: #1E293B; border: 1px solid #334155; padding: 12px 18px; border-radius: 10px; min-width: 140px;">
                <div style="color: #94A3B8; font-size: 10px; font-weight: 600;">STAGE 1</div>
                <div style="color: #F8FAFC; font-size: 14px; font-weight: 700; margin-top: 2px;">🌐 CURRENT NETWORK</div>
            </div>
            <div style="color: #38BDF8; font-size: 18px; font-weight: 800;">➔</div>
            <div style="background: #1E293B; border: 1px solid #0EA5E9; padding: 12px 18px; border-radius: 10px; min-width: 140px;">
                <div style="color: #94A3B8; font-size: 10px; font-weight: 600;">STAGE 2</div>
                <div style="color: #38BDF8; font-size: 14px; font-weight: 700; margin-top: 2px;">🪞 DIGITAL TWIN</div>
            </div>
            <div style="color: #38BDF8; font-size: 18px; font-weight: 800;">➔</div>
            <div style="background: #1E293B; border: 1px solid #A78BFA; padding: 12px 18px; border-radius: 10px; min-width: 140px;">
                <div style="color: #94A3B8; font-size: 10px; font-weight: 600;">STAGE 3</div>
                <div style="color: #A78BFA; font-size: 14px; font-weight: 700; margin-top: 2px;">🔮 LSTM PREDICTION</div>
            </div>
            <div style="color: #38BDF8; font-size: 18px; font-weight: 800;">➔</div>
            <div style="background: #1E293B; border: 1px solid #F59E0B; padding: 12px 18px; border-radius: 10px; min-width: 140px;">
                <div style="color: #94A3B8; font-size: 10px; font-weight: 600;">STAGE 4</div>
                <div style="color: #F59E0B; font-size: 14px; font-weight: 700; margin-top: 2px;">🎛️ ADAPTIVE CONTROLLER</div>
            </div>
            <div style="color: #38BDF8; font-size: 18px; font-weight: 800;">➔</div>
            <div style="background: #1E293B; border: 1px solid #10B981; padding: 12px 18px; border-radius: 10px; min-width: 160px;">
                <div style="color: #94A3B8; font-size: 10px; font-weight: 600;">STAGE 5</div>
                <div style="color: #10B981; font-size: 14px; font-weight: 700; margin-top: 2px;">📡 COMMUNICATION PROFILE</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --- EDGE NODE SELECTOR ---
node_options = [f"N{i}" for i in range(1, 11)]
curr_selected = st.session_state.get("selected_node", "N1")
node_idx = node_options.index(curr_selected) if curr_selected in node_options else 0

selected_node = st.selectbox(
    "Select Target Edge Node for Adaptive Control Evaluation:",
    node_options,
    index=node_idx,
    key="p7_node_select"
)
st.session_state["selected_node"] = selected_node

# Fetch REAL Telemetry, PyTorch LSTM Forecast, and Adaptive Controller Output
target_telemetry = dt_engine.state.node_telemetry.get(selected_node, {})
pred_res = predictor.predict_node_future_snr(dt_engine.state, selected_node)
profile_res = controller.select_profile(pred_res["predicted_future_snr"])

current_snr_val = float(pred_res["current_snr"])
predicted_snr_val = float(pred_res["predicted_future_snr"])
current_cond_val = str(target_telemetry.get("health", "GOOD")).upper()
predicted_cond_val = str(pred_res["predicted_condition"]).upper()
controller_decision = profile_res["mode"]
selected_profile = profile_res["profile"]

is_high_fid = (controller_decision == "HIGH_FIDELITY")
border_color = "#10B981" if is_high_fid else "#F59E0B"
mode_badge = "🟢 HIGH FIDELITY" if is_high_fid else "🟡 HIGH COMPRESSION (ROBUST)"
pred_cond_color = "#10B981" if predicted_cond_val == "GOOD" else ("#F59E0B" if predicted_cond_val in ["MODERATE", "DEGRADED"] else "#EF4444")
cond_emoji = "🟢" if predicted_cond_val == "GOOD" else ("🟡" if predicted_cond_val in ["MODERATE", "DEGRADED"] else "🔴")

# --- 2. PROMINENT CONTROLLER DECISION HERO CARD ---
st.markdown("#### 🎯 PROACTIVE CONTROLLER DECISION")

st.markdown(
    f"""
    <div style="background-color: #0F172A; border: 2px solid {border_color}; border-radius: 14px; padding: 24px; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <div>
                <span style="color: #94A3B8; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;">CONTROLLER DECISION MODE</span>
                <div style="font-size: 26px; font-weight: 800; color: #F8FAFC; margin-top: 4px;">
                    <span style="color:{border_color};">{mode_badge}</span>
                </div>
            </div>
            <div style="background: #1E293B; border: 1px solid {border_color}; padding: 10px 20px; border-radius: 10px; text-align: right;">
                <span style="color: #94A3B8; font-size: 11px; font-weight: 600;">ALLOCATED CHANNELS</span>
                <div style="color: #38BDF8; font-size: 20px; font-weight: 800;">{profile_res['channels']} Channels</div>
            </div>
        </div>
        <hr style="border-color: #1E293B; margin: 14px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="color: #94A3B8; font-size: 12px; font-weight: 600;">SELECTED COMMUNICATION PROFILE</span>
                <div style="font-size: 22px; font-weight: 800; color: #FFD700; margin-top: 2px;">
                    📡 {selected_profile}
                </div>
            </div>
            <div style="max-width: 55%; color: #94A3B8; font-size: 13px; line-height: 1.5; text-align: right;">
                💡 <b>Control Rationale:</b> {profile_res['description']}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --- 3. FIVE METRIC CARDS REQUIRED ---
st.markdown(f"#### 📊 Adaptive Control Telemetry & Metrics (`{selected_node}`)")

st.markdown(
    f"""
    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 24px;">
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 10px; padding: 16px; text-align: center;">
            <span style="color: #94A3B8; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;">CURRENT CONDITION</span>
            <div style="color: #F8FAFC; font-size: 18px; font-weight: 700; margin-top: 6px;">{current_cond_val}</div>
        </div>
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 10px; padding: 16px; text-align: center;">
            <span style="color: #94A3B8; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;">PREDICTED CONDITION</span>
            <div style="color: {pred_cond_color}; font-size: 18px; font-weight: 700; margin-top: 6px;">{cond_emoji} {predicted_cond_val}</div>
        </div>
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 10px; padding: 16px; text-align: center;">
            <span style="color: #94A3B8; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;">CURRENT SNR</span>
            <div style="color: #F8FAFC; font-size: 18px; font-weight: 700; margin-top: 6px;">{current_snr_val:.1f} dB</div>
        </div>
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 10px; padding: 16px; text-align: center;">
            <span style="color: #94A3B8; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;">PREDICTED SNR</span>
            <div style="color: #38BDF8; font-size: 18px; font-weight: 700; margin-top: 6px;">{predicted_snr_val:.1f} dB</div>
        </div>
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 10px; padding: 16px; text-align: center;">
            <span style="color: #94A3B8; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;">SELECTED PROFILE</span>
            <div style="color: #FFD700; font-size: 16px; font-weight: 700; margin-top: 6px;">{selected_profile}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --- CONTROLS & NAVIGATION ---
col_ac1, col_ac2 = st.columns(2)
with col_ac1:
    if st.button("🎛️ Re-Evaluate Proactive Adaptive Control", key="p7_eval_ctrl_btn", use_container_width=True, type="primary"):
        st.rerun()

with col_ac2:
    if st.button("Continue → Results", key="btn_continue_to_results_p7", use_container_width=True, type="primary"):
        try:
            st.switch_page("pages/8_Results.py")
        except Exception:
            st.info("👈 Select **📊 Results** in the sidebar navigation to proceed.")
