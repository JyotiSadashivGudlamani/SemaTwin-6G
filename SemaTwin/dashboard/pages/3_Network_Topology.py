import streamlit as st
import os
import sys
import plotly.graph_objects as go

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from digital_twin.twin import DigitalTwinEngine
from dashboard.ui_components import render_dashboard_header

st.set_page_config(page_title="3. 🌐 Network Topology", page_icon="🌐", layout="wide")

render_dashboard_header(current_stage="Network")

# Initialize DigitalTwinEngine in session state to retrieve existing node status
if "dt_engine" not in st.session_state or not hasattr(st.session_state["dt_engine"], "state"):
    st.session_state["dt_engine"] = DigitalTwinEngine()

dt_engine = st.session_state["dt_engine"]
node_telemetry = dt_engine.state.node_telemetry

# Shared experiment semantic meaning from session_state
sem_meaning = st.session_state.get("semantic_meaning", st.session_state.get("actual_class", "DOG")).upper()

# Ensure default selected nodes list in session_state
if "selected_nodes" not in st.session_state or not st.session_state["selected_nodes"]:
    st.session_state["selected_nodes"] = ["N1", "N3", "N7"]

st.markdown("### 🌐 6G Network Topology & Architecture")
st.caption("Interactive network structure: Server 1 (N1–N5) and Server 2 (N6–N10)")

# --- SEMANTIC MEANING TO TRANSMIT ---
st.markdown(
    f"""
    <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="color: #94A3B8; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;">SEMANTIC TO TRANSMIT</span>
            <div style="color: #38BDF8; font-size: 20px; font-weight: 700; margin-top: 2px;">📦 {sem_meaning}</div>
        </div>
        <div style="color: #64748B; font-size: 24px; font-weight: 700;">➔</div>
        <div>
            <span style="color: #94A3B8; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;">TARGET DESTINATIONS</span>
            <div style="color: #F59E0B; font-size: 18px; font-weight: 700; margin-top: 2px;">{', '.join(st.session_state['selected_nodes']) if st.session_state['selected_nodes'] else 'None'}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- MULTI-NODE SELECTION CONTROLS ---
node_options = [f"N{i}" for i in range(1, 11)]
selected_nodes = st.multiselect(
    "Select Destination Nodes (Multi-Node Routing):",
    node_options,
    default=[n for n in st.session_state["selected_nodes"] if n in node_options] if st.session_state["selected_nodes"] else ["N1", "N3", "N7"],
    key="p3_multiselect_nodes"
)

if not selected_nodes:
    selected_nodes = ["N1"]

st.session_state["selected_nodes"] = selected_nodes
if selected_nodes:
    st.session_state["selected_node"] = selected_nodes[0]

# --- INTERACTIVE TOPOLOGY ARCHITECTURE DIAGRAM (PLOTLY) ---
server_x = [-2.5, 2.5]
server_y = [2.0, 2.0]

s1_nodes = [f"N{i}" for i in range(1, 6)]
s2_nodes = [f"N{i}" for i in range(6, 11)]

device_x, device_y = [], []
device_colors, device_sizes, device_symbols, device_labels, hover_texts = [], [], [], [], []

# Server 1 Devices (N1..N5)
for idx, nid in enumerate(s1_nodes):
    x_pos = -4.5 + idx * 1.0
    device_x.append(x_pos)
    device_y.append(0.5)
    
    is_sel = (nid in selected_nodes)
    status_str = node_telemetry.get(nid, {}).get("status", "ACTIVE").upper()
    status_icon = "● ACTIVE" if status_str != "INACTIVE" else "● INACTIVE"
    
    device_colors.append("#F59E0B" if is_sel else "#38BDF8")
    device_sizes.append(34 if is_sel else 24)
    device_symbols.append("star" if is_sel else "circle")
    device_labels.append(f"<b>🎯 {nid}</b>" if is_sel else nid)
    hover_texts.append(f"Node: {nid}<br>Server: Server 1<br>Status: {status_icon}")

# Server 2 Devices (N6..N10)
for idx, nid in enumerate(s2_nodes):
    x_pos = 0.5 + idx * 1.0
    device_x.append(x_pos)
    device_y.append(0.5)
    
    is_sel = (nid in selected_nodes)
    status_str = node_telemetry.get(nid, {}).get("status", "ACTIVE").upper()
    status_icon = "● ACTIVE" if status_str != "INACTIVE" else "● INACTIVE"
    
    device_colors.append("#F59E0B" if is_sel else "#38BDF8")
    device_sizes.append(34 if is_sel else 24)
    device_symbols.append("star" if is_sel else "circle")
    device_labels.append(f"<b>🎯 {nid}</b>" if is_sel else nid)
    hover_texts.append(f"Node: {nid}<br>Server: Server 2<br>Status: {status_icon}")

# Edge lines connecting Server 1 -> N1..N5 and Server 2 -> N6..N10
edge_x, edge_y = [], []
for idx in range(5):
    edge_x.extend([-2.5, device_x[idx], None])
    edge_y.extend([2.0, 0.5, None])
for idx in range(5):
    edge_x.extend([2.5, device_x[5 + idx], None])
    edge_y.extend([2.0, 0.5, None])

trace_edges = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=2, color="#334155"),
    hoverinfo="none", mode="lines",
    showlegend=False
)

trace_servers = go.Scatter(
    x=server_x, y=server_y,
    mode="markers+text",
    text=["SERVER 1\n(N1–N5)", "SERVER 2\n(N6–N10)"],
    textposition="top center",
    textfont=dict(size=12, color="#F8FAFC", family="Arial Black"),
    hoverinfo="text",
    hovertext=["Server 1 (Hosts N1-N5)", "Server 2 (Hosts N6-N10)"],
    marker=dict(
        color=["#0EA5E9", "#0EA5E9"],
        size=36,
        symbol="square",
        line=dict(width=2, color="#FFFFFF")
    ),
    showlegend=False
)

trace_devices = go.Scatter(
    x=device_x, y=device_y,
    mode="markers+text",
    text=device_labels,
    textposition="bottom center",
    textfont=dict(size=12, color="#F8FAFC"),
    hoverinfo="text",
    hovertext=hover_texts,
    marker=dict(
        color=device_colors,
        size=device_sizes,
        symbol=device_symbols,
        line=dict(width=2, color="#FFFFFF")
    ),
    showlegend=False
)

fig_net = go.Figure(data=[trace_edges, trace_servers, trace_devices])
fig_net.update_layout(
    template="plotly_dark",
    hovermode="closest",
    margin=dict(b=30, l=20, r=20, t=30),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5.2, 5.2]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.2, 2.5]),
    height=320,
    paper_bgcolor="#0F172A",
    plot_bgcolor="#0F172A"
)

st.plotly_chart(fig_net, use_container_width=True)

st.markdown("---")

# --- NAVIGATION CONTINUATION BUTTON ---
if st.button("Continue → Communication", key="btn_continue_to_comm_p3", use_container_width=True, type="primary"):
    try:
        st.switch_page("pages/4_Communication.py")
    except Exception:
        st.info("👈 Select **📡 Communication** in the sidebar navigation to proceed.")

