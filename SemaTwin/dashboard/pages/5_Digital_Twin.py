import streamlit as st
import os
import sys
import json
import plotly.graph_objects as go

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from digital_twin.twin import DigitalTwinEngine
from dashboard.ui_components import render_dashboard_header
from dashboard.dt_3d import render_3d_digital_twin

st.set_page_config(page_title="5. 🪞 Digital Twin", page_icon="🪞", layout="wide")

render_dashboard_header(current_stage="Digital Twin")

if "dt_engine" not in st.session_state or not hasattr(st.session_state["dt_engine"], "state"):
    st.session_state["dt_engine"] = DigitalTwinEngine()

dt_engine = st.session_state["dt_engine"]
node_telemetry = dt_engine.state.node_telemetry

if "selected_node" not in st.session_state or not st.session_state["selected_node"]:
    st.session_state["selected_node"] = "N3"

selected_dest_nodes = st.session_state.get("selected_nodes", ["N1", "N3", "N7"])

st.markdown("### 🪞 6G Network Digital Twin & Virtual Mirror")
st.caption("Live 3D spatial simulation and real-time telemetry mirror (Server 1: N1–N5 | Server 2: N6–N10)")

# --- 1. 3D DIGITAL TWIN VISUALIZATION ---
st.markdown("#### 🧊 3D Interactive Network Topology Twin")
render_3d_digital_twin(node_telemetry, selected_dest_nodes, height=450)

# Helper function to compute health category & color
def evaluate_node_health(t_data):
    st_val = t_data.get("status", "ACTIVE").upper()
    h_val = t_data.get("health", "GOOD").upper()
    loss = float(t_data.get("packet_loss_pct", 0.0))
    snr = float(t_data.get("snr_db", 15.0))
    
    if st_val == "INACTIVE":
        return "INACTIVE", "#64748B"
    elif h_val in ["EXCELLENT", "GOOD"] and loss < 3.0 and snr >= 12.0:
        return "GOOD", "#10B981"
    elif h_val == "DEGRADED" or (3.0 <= loss < 8.0) or (7.0 <= snr < 12.0):
        return "DEGRADED", "#F59E0B"
    else:
        return "POOR", "#EF4444"

# --- 2. SELECTED NODE TELEMETRY PANEL ---
st.markdown("---")
st.markdown("### 📋 SELECTED NODE TELEMETRY")

node_options = [f"N{i}" for i in range(1, 11)]
curr_inspect = st.session_state.get("selected_node", "N3")
inspect_idx = node_options.index(curr_inspect) if curr_inspect in node_options else 2

selected_inspect_node = st.selectbox(
    "Select Node to Inspect Telemetry:",
    node_options,
    index=inspect_idx,
    key="p5_inspect_node_select"
)

st.session_state["selected_node"] = selected_inspect_node

# Fetch existing telemetry from backend
t_inspect = node_telemetry.get(selected_inspect_node, {})
insp_snr = float(t_inspect.get("snr_db", 15.0))
insp_lat = float(t_inspect.get("latency_ms", 25.0))
insp_bw = float(t_inspect.get("bandwidth_mbps", 72.0))
insp_loss = float(t_inspect.get("packet_loss_pct", 0.0))
insp_server = "Server 1" if int(selected_inspect_node.replace("N", "")) <= 5 else "Server 2"
insp_health, insp_color = evaluate_node_health(t_inspect)

st.markdown(
    f"""
    <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div>
                <span style="color: #94A3B8; font-size: 12px; font-weight: 600;">SELECTED NODE</span>
                <div style="color: #38BDF8; font-size: 24px; font-weight: 700;">{selected_inspect_node}</div>
            </div>
            <div>
                <span style="color: #94A3B8; font-size: 12px; font-weight: 600;">PARENT SERVER</span>
                <div style="color: #F8FAFC; font-size: 18px; font-weight: 700;">{insp_server}</div>
            </div>
            <div>
                <span style="color: #94A3B8; font-size: 12px; font-weight: 600;">HEALTH STATUS</span>
                <div style="color: {insp_color}; background: rgba(15, 23, 42, 0.8); padding: 4px 14px; border-radius: 8px; border: 1px solid {insp_color}; font-size: 16px; font-weight: 700; text-align: center; margin-top: 2px;">
                    ● {insp_health}
                </div>
            </div>
        </div>
        <hr style="border-color: #1E293B; margin: 12px 0;">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; text-align: center;">
            <div style="background: #1E293B; border-radius: 8px; padding: 12px;">
                <span style="color: #94A3B8; font-size: 11px; font-weight: 600;">SNR</span>
                <div style="color: #F8FAFC; font-size: 20px; font-weight: 700; margin-top: 4px;">{insp_snr:.1f} dB</div>
            </div>
            <div style="background: #1E293B; border-radius: 8px; padding: 12px;">
                <span style="color: #94A3B8; font-size: 11px; font-weight: 600;">LATENCY</span>
                <div style="color: #F8FAFC; font-size: 20px; font-weight: 700; margin-top: 4px;">{insp_lat:.1f} ms</div>
            </div>
            <div style="background: #1E293B; border-radius: 8px; padding: 12px;">
                <span style="color: #94A3B8; font-size: 11px; font-weight: 600;">BANDWIDTH</span>
                <div style="color: #F8FAFC; font-size: 20px; font-weight: 700; margin-top: 4px;">{insp_bw:.1f} Mbps</div>
            </div>
            <div style="background: #1E293B; border-radius: 8px; padding: 12px;">
                <span style="color: #94A3B8; font-size: 11px; font-weight: 600;">PACKET LOSS</span>
                <div style="color: {insp_color}; font-size: 20px; font-weight: 700; margin-top: 4px;">{insp_loss:.1f}%</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- 3. CHANNEL CONDITION SIMULATION CONTROLS ---
st.markdown("---")
st.markdown("### 🎛️ Digital Twin Telemetry Controls")
col_scen, col_act = st.columns([3, 1])

with col_scen:
    scen_choice = st.radio(
        "Select Telemetry Simulation Scenario:",
        ["normal", "degrading", "high_interference", "random_fluctuation"],
        format_func=lambda x: {
            "normal": "🟢 Normal Conditions (High SNR, Low Latency, High Bandwidth)",
            "degrading": "🟡 Degrading Channel (Fading / Distance Increase)",
            "high_interference": "🔴 High Interference (Low SNR, High Latency, High Packet Loss)",
            "random_fluctuation": "🎲 Random Network Fluctuation"
        }[x],
        horizontal=True,
        key="p5_telemetry_scenario"
    )

with col_act:
    st.write("")
    st.write("")
    if st.button("🔄 Sync / Update Telemetry", key="page5_update_telemetry_btn", use_container_width=True, type="primary"):
        dt_engine.update_telemetry(scenario=scen_choice)
        st.rerun()

st.markdown("---")

# --- NAVIGATION CONTINUATION BUTTON ---
if st.button("Continue → Prediction", key="btn_continue_to_pred_p5", use_container_width=True, type="primary"):
    try:
        st.switch_page("pages/6_Prediction.py")
    except Exception:
        st.info("👈 Select **🔮 Prediction** in the sidebar navigation to proceed.")
