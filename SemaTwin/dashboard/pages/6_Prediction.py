import streamlit as st
import os
import sys
import plotly.graph_objects as go

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib
import digital_twin.state
import digital_twin.twin
import digital_twin.predictor
importlib.reload(digital_twin.state)
importlib.reload(digital_twin.twin)
importlib.reload(digital_twin.predictor)

from digital_twin.twin import DigitalTwinEngine
from digital_twin.predictor import DigitalTwinPredictor
from dashboard.ui_components import render_dashboard_header

st.set_page_config(page_title="6. 🔮 Prediction", page_icon="🔮", layout="wide")

render_dashboard_header(current_stage="Prediction")

if "dt_engine" not in st.session_state:
    st.session_state["dt_engine"] = DigitalTwinEngine()

dt_engine = st.session_state["dt_engine"]
predictor = DigitalTwinPredictor()

st.markdown("### 🔮 6G Wireless Channel AI Prediction & Forecast")
st.caption("PyTorch LSTM Neural Network forecasting future wireless channel conditions and SNR trajectories")

# --- EDGE NODE SELECTOR ---
node_options = [f"N{i}" for i in range(1, 11)]
curr_selected = st.session_state.get("selected_node", "N1")
node_idx = node_options.index(curr_selected) if curr_selected in node_options else 0

selected_node = st.selectbox(
    "Select Destination Node to Inspect LSTM Forecast:",
    node_options,
    index=node_idx,
    key="p6_node_select"
)
st.session_state["selected_node"] = selected_node

# Fetch REAL PyTorch LSTM Forecast for selected node
pred_res = predictor.predict_node_future_snr(dt_engine.state, selected_node)
st.session_state["last_pred_res"] = pred_res

current_snr_val = float(pred_res["current_snr"])
predicted_snr_val = float(pred_res["predicted_future_snr"])
hist_snr = list(pred_res["historical_snr"])

snr_delta = predicted_snr_val - current_snr_val

if snr_delta < -0.5:
    net_condition = "DEGRADING"
    cond_color = "#EF4444"
elif snr_delta > 0.5:
    net_condition = "IMPROVING"
    cond_color = "#10B981"
else:
    net_condition = "STABLE"
    cond_color = "#38BDF8"

# --- 1. THREE PROMINENT METRIC CARDS ---
st.markdown("---")
st.markdown(f"#### 📊 Real-Time LSTM Telemetry Metrics (`{selected_node}`)")

st.markdown(
    f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px;">
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 12px; padding: 20px; text-align: center;">
            <span style="color: #94A3B8; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;">CURRENT SNR</span>
            <div style="color: #F8FAFC; font-size: 28px; font-weight: 800; margin-top: 6px;">{current_snr_val:.1f} dB</div>
            <span style="color: #64748B; font-size: 11px;">Measured at {selected_node}</span>
        </div>
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 12px; padding: 20px; text-align: center;">
            <span style="color: #94A3B8; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;">PREDICTED SNR</span>
            <div style="color: #38BDF8; font-size: 28px; font-weight: 800; margin-top: 6px;">{predicted_snr_val:.1f} dB</div>
            <span style="color: {cond_color}; font-size: 12px; font-weight: 600;">{snr_delta:+.1f} dB forecast</span>
        </div>
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 12px; padding: 20px; text-align: center;">
            <span style="color: #94A3B8; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;">NETWORK CONDITION</span>
            <div style="color: {cond_color}; font-size: 24px; font-weight: 800; margin-top: 6px; letter-spacing: 0.5px;">
                ● {net_condition}
            </div>
            <span style="color: #94A3B8; font-size: 11px;">LSTM Channel Evaluation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --- 2. PLOTLY FORECAST VISUALIZATION (SNR HISTORY -> FUTURE SNR) ---
st.markdown("#### 📈 SNR HISTORY → FUTURE SNR")
st.caption(f"Historical channel trajectory vs. PyTorch LSTM predicted future point for node `{selected_node}`")

# X-axis time labels
hist_steps = [f"T-{len(hist_snr)-1-i}" for i in range(len(hist_snr))]
future_step = "T+2s (Forecast)"

# Create Plotly figure with clearly distinguished traces
fig_snr = go.Figure()

# Trace 1: Historical SNR (Solid Line & Markers)
fig_snr.add_trace(go.Scatter(
    x=hist_steps,
    y=hist_snr,
    mode="lines+markers",
    name="Historical SNR",
    line=dict(color="#38BDF8", width=3),
    marker=dict(size=8, color="#0EA5E9", symbol="circle")
))

# Trace 2: Predicted SNR (Dashed Line connecting T-0 to Forecast Point)
fig_snr.add_trace(go.Scatter(
    x=[hist_steps[-1], future_step],
    y=[hist_snr[-1], predicted_snr_val],
    mode="lines+markers+text",
    name="Predicted SNR (LSTM)",
    line=dict(color=cond_color, width=3, dash="dash"),
    marker=dict(size=14, color=cond_color, symbol="star"),
    text=["", f"<b>{predicted_snr_val:.1f} dB</b>"],
    textposition="top center",
    textfont=dict(color="#F8FAFC", size=13)
))

# Vertical line separating history from forecast horizon
fig_snr.add_vline(
    x=hist_steps[-1],
    line_width=1,
    line_dash="dot",
    line_color="#64748B",
    annotation_text="Present Time (T-0)",
    annotation_position="top left",
    annotation_font=dict(color="#94A3B8", size=11)
)

fig_snr.update_layout(
    template="plotly_dark",
    height=380,
    margin=dict(l=20, r=20, t=40, b=20),
    paper_bgcolor="#0F172A",
    plot_bgcolor="#0F172A",
    xaxis=dict(title="Time Step (Telemetry Window)", showgrid=True, gridcolor="#1E293B"),
    yaxis=dict(title="SNR (dB)", showgrid=True, gridcolor="#1E293B"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_snr, use_container_width=True)

st.markdown("---")

# --- CONTROLS & NAVIGATION ---
col_p1, col_p2 = st.columns(2)
with col_p1:
    if st.button("🔮 Refresh LSTM Forecast", key="p6_refresh_lstm_btn", use_container_width=True, type="primary"):
        st.rerun()

with col_p2:
    if st.button("Continue → Adaptive Control", key="btn_continue_to_ctrl_p6", use_container_width=True, type="primary"):
        try:
            st.switch_page("pages/7_Adaptive_Control.py")
        except Exception:
            st.info("👈 Select **🎛️ Adaptive Control** in the sidebar navigation to proceed.")
