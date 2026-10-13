import os
import sys

# Windows PyTorch / OpenMP DLL Initialization fix (WinError 1114 mitigation)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

if sys.platform == "win32":
    try:
        torch_lib_path = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
        if os.path.exists(torch_lib_path):
            os.add_dll_directory(torch_lib_path)
    except Exception:
        pass

import streamlit as st
import numpy as np
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from streamlit_option_menu import option_menu

# Ensure PROJECT_ROOT is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import data.dataset
from data.dataset import CIFAR10DataModule, CIFAR10_CLASSES
from models.resnet_semantic import ResNet18CIFAR10
from semantic.feature_extractor import ResNet18SemanticFeatureExtractor

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

from dashboard.ui_components import render_dashboard_header
from dashboard.styles import get_custom_css, render_footer

# --- STAGE CONFIGURATION ---
st.set_page_config(
    page_title="6G Semantic Communication & Digital Twin Control Center",
    page_icon="🛰️",
    layout="wide"
)

# Inject Custom 6G Dark Tech CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# --- STATE-BASED NAVIGATION SETUP ---
PAGE_LIST = [
    "Overview",
    "Image & Semantics",
    "Network",
    "Communication",
    "Digital Twin",
    "Prediction",
    "Adaptive Control",
    "Results",
    "About"
]

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Overview"

# Handle pending navigation requests from action buttons
if "pending_nav_section" in st.session_state:
    raw_pending = st.session_state.pop("pending_nav_section")
    clean_pending = raw_pending.replace("🏠 ", "").replace("🖼️ ", "").replace("🌐 ", "").replace("📡 ", "").replace("🪞 ", "").replace("🔮 ", "").replace("🎛️ ", "").replace("📊 ", "").replace("ℹ️ ", "").strip()
    if clean_pending in PAGE_LIST:
        st.session_state["current_page"] = clean_pending
        st.session_state["app_sidebar_option_menu"] = clean_pending

current_page = st.session_state.get("current_page", "Overview")
if current_page not in PAGE_LIST:
    current_page = "Overview"
    st.session_state["current_page"] = "Overview"
    st.session_state["app_sidebar_option_menu"] = "Overview"

def navigate_to(page_name):
    clean_name = page_name.replace("🏠 ", "").replace("🖼️ ", "").replace("🌐 ", "").replace("📡 ", "").replace("🪞 ", "").replace("🔮 ", "").replace("🎛️ ", "").replace("📊 ", "").replace("ℹ️ ", "").strip()
    if clean_name in PAGE_LIST:
        st.session_state["pending_nav_section"] = clean_name
        st.session_state["current_page"] = clean_name
        st.rerun()

# --- SIDEBAR NAVIGATION WITH OPTION MENU (STATIC KEY FOR INSTANT SWITCHING) ---
with st.sidebar:
    st.title("🛰️ Digital Twin 6G")
    st.caption("Semantic Communication System")
    
    icons = [
        "house",
        "image",
        "diagram-3",
        "broadcast",
        "cpu",
        "graph-up",
        "sliders",
        "bar-chart",
        "info-circle"
    ]
    
    default_idx = PAGE_LIST.index(current_page)

    selected_menu_item = option_menu(
        menu_title=None,
        options=PAGE_LIST,
        icons=icons,
        default_index=default_idx,
        manual_select=default_idx,
        key="app_sidebar_option_menu",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#38BDF8", "font-size": "15px"},
            "nav-link": {
                "font-size": "13px",
                "text-align": "left",
                "margin": "3px 0px",
                "--hover-color": "#1E293B",
                "color": "#94A3B8",
                "font-weight": "500"
            },
            "nav-link-selected": {
                "background-color": "#1E293B",
                "color": "#38BDF8",
                "font-weight": "600",
                "border-left": "3px solid #38BDF8"
            },
        }
    )
    
    if selected_menu_item != current_page:
        st.session_state["current_page"] = selected_menu_item
        st.rerun()


# --- CACHED MODEL & DATA INITIALIZATION ---
@st.cache_resource
def get_data_module():
    return data.dataset.CIFAR10DataModule()

@st.cache_resource
def get_resnet_model():
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

@st.cache_resource
def get_digital_twin_predictor():
    from digital_twin.predictor import DigitalTwinPredictor
    return DigitalTwinPredictor()

@st.cache_resource
def get_adaptive_controller():
    from control.adaptive_controller import ProactiveAdaptiveController
    return ProactiveAdaptiveController()

def get_transmission_pipeline():
    import importlib
    import communication.transmission
    importlib.reload(communication.transmission)
    return communication.transmission.SemanticTransmissionPipeline()

def update_shared_experiment_image(pil_img, cls_name, img_path, model, extractor):
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

def ensure_shared_experiment_image():
    if "selected_image" not in st.session_state or st.session_state["selected_image"] is None:
        try:
            dm = get_data_module()
            model = get_resnet_model()
            extractor = get_feature_extractor()
            pil_img, cls_name, _, _, img_path = dm.get_correctly_classified_sample_by_class("dog", model=model)
            update_shared_experiment_image(pil_img, cls_name, img_path, model, extractor)
        except Exception as e:
            print(f"[EXPERIMENT IMG INIT WARNING] {e}")

# Global initialization once
ensure_shared_experiment_image()


# ====================================================================
# PAGE 1: OVERVIEW
# ====================================================================
def render_overview():
    render_dashboard_header(current_stage="Overview")

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

    st.markdown("### 🚀 Start Experiment Pipeline")
    st.caption("Begin the continuous experiment sequence: Image & Semantics ➔ Network ➔ Communication ➔ Digital Twin ➔ Prediction ➔ Control ➔ Results.")
    if st.button("🚀 Start Experiment ➔ Go to Image & Semantics", key="btn_start_exp_app", use_container_width=True, type="primary"):
        navigate_to("Image & Semantics")


# ====================================================================
# PAGE 2: IMAGE & SEMANTICS
# ====================================================================
def render_image_semantics():
    render_dashboard_header(current_stage="Image & Semantics")

    dm = get_data_module()
    model = get_resnet_model()
    extractor = get_feature_extractor()

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.markdown("### 🖼️ Selected CIFAR-10 Image")
        selected_class = st.selectbox(
            "Select CIFAR-10 Class Category:",
            CIFAR10_CLASSES,
            index=CIFAR10_CLASSES.index(st.session_state.get("actual_class", "dog").lower()) if st.session_state.get("actual_class", "dog").lower() in CIFAR10_CLASSES else 5,
            key="task3_exp_cls_select"
        )
        if "last_selected_exp_class" not in st.session_state or st.session_state.get("last_selected_exp_class") != selected_class:
            pil_img, cls_name, cls_idx, img_np, img_path = dm.get_correctly_classified_sample_by_class(selected_class, model=model)
            update_shared_experiment_image(pil_img, cls_name, img_path, model, extractor)
            st.session_state["last_selected_exp_class"] = selected_class

        if st.button("🎲 Fetch Another Sample Image", key="task3_exp_fetch_btn", use_container_width=True):
            pil_img, cls_name, cls_idx, img_np, img_path = dm.get_correctly_classified_sample_by_class(selected_class, model=model)
            update_shared_experiment_image(pil_img, cls_name, img_path, model, extractor)

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
    if st.button("Continue → Network Topology", key="btn_continue_to_net_app", use_container_width=True, type="primary"):
        navigate_to("Network")


# ====================================================================
# PAGE 3: NETWORK TOPOLOGY
# ====================================================================
def render_network():
    render_dashboard_header(current_stage="Network")

    from digital_twin.twin import DigitalTwinEngine

    if "dt_engine" not in st.session_state or not hasattr(st.session_state["dt_engine"], "update_telemetry"):
        st.session_state["dt_engine"] = DigitalTwinEngine()

    dt_engine = st.session_state["dt_engine"]
    node_telemetry = dt_engine.state.node_telemetry

    sem_meaning = st.session_state.get("semantic_meaning", st.session_state.get("actual_class", "DOG")).upper()

    if "selected_nodes" not in st.session_state or not st.session_state["selected_nodes"]:
        st.session_state["selected_nodes"] = ["N1", "N3", "N7"]

    st.markdown("### 🌐 6G Network Topology & Architecture")
    st.caption("Interactive network structure: Server 1 (N1–N5) and Server 2 (N6–N10)")

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

    node_options = [f"N{i}" for i in range(1, 11)]
    selected_nodes = st.multiselect(
        "Select Destination Nodes (Multi-Node Routing):",
        node_options,
        default=[n for n in st.session_state["selected_nodes"] if n in node_options] if st.session_state["selected_nodes"] else ["N1", "N3", "N7"],
        key="app_p3_multiselect_nodes"
    )

    if not selected_nodes:
        selected_nodes = ["N1"]

    st.session_state["selected_nodes"] = selected_nodes
    if selected_nodes:
        st.session_state["selected_node"] = selected_nodes[0]

    server_x = [-2.5, 2.5]
    server_y = [2.0, 2.0]

    s1_nodes = [f"N{i}" for i in range(1, 6)]
    s2_nodes = [f"N{i}" for i in range(6, 11)]

    device_x, device_y = [], []
    device_colors, device_sizes, device_symbols, device_labels, hover_texts = [], [], [], [], []

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
    if st.button("Continue → Communication", key="btn_continue_to_comm_app", use_container_width=True, type="primary"):
        navigate_to("Communication")


# ====================================================================
# PAGE 4: COMMUNICATION & TRANSMISSION
# ====================================================================
def render_communication():
    render_dashboard_header(current_stage="Communication")

    from digital_twin.twin import DigitalTwinEngine

    if "dt_engine" not in st.session_state:
        st.session_state["dt_engine"] = DigitalTwinEngine()

    dt_engine = st.session_state["dt_engine"]
    pipeline = get_transmission_pipeline()

    selected_target_nodes = st.session_state.get("selected_nodes", ["N1", "N3", "N7"])
    if not selected_target_nodes:
        selected_target_nodes = ["N1"]

    primary_node = selected_target_nodes[0]
    st.session_state["selected_node"] = primary_node

    curr_img_comm = st.session_state["selected_image"]
    comm_class = st.session_state["actual_class"]
    sem_meaning = st.session_state.get("semantic_meaning", comm_class.upper()).upper()

    servers_set = set()
    for nid in selected_target_nodes:
        servers_set.add(pipeline.determine_source_server(nid))
    servers_str = ", ".join(sorted(list(servers_set)))
    nodes_str = ", ".join(selected_target_nodes)

    primary_t = dt_engine.state.node_telemetry.get(primary_node, {})
    primary_snr = primary_t.get("snr_db", 15.0)
    primary_lat = primary_t.get("latency_ms", 25.0)
    primary_bw = primary_t.get("bandwidth_mbps", 100.0)
    curr_snr_val = float(st.session_state.get("custom_snr_db", primary_snr))
    primary_loss = max(0.0, min(100.0, (20.0 - curr_snr_val) * 4.25))

    if "multi_node_results" in st.session_state and st.session_state["multi_node_results"]:
        res_list = st.session_state["multi_node_results"]
        tot_sent = sum(r.get("packets_sent", 100) for r in res_list)
        tot_lost = sum(r.get("packets_lost", 0) for r in res_list)
        if tot_sent > 0:
            primary_loss = (tot_lost / tot_sent) * 100.0

    st.markdown("### 📡 6G Transmission Monitoring Interface")
    st.caption("End-to-End Deep JSCC Semantic Transmission & Real-Time Monitoring")

    col_sel1, col_sel2 = st.columns([1.2, 1])

    with col_sel1:
        sel_channel_label = st.selectbox(
            "🌐 Select 6G Channel Model:",
            ["AWGN Channel (Additive White Gaussian Noise)", "Rayleigh Fading Channel (Multipath Fading)"],
            index=0 if st.session_state.get("comm_channel_type", "AWGN") == "AWGN" else 1,
            key="app_p5_channel_type_select"
        )
        channel_type = "Rayleigh" if "Rayleigh" in sel_channel_label else "AWGN"
        st.session_state["comm_channel_type"] = channel_type

    with col_sel2:
        init_snr = int(st.session_state.get("custom_snr_db", int(primary_snr)))
        init_snr = min(20, max(0, init_snr))
        selected_snr_db = st.slider(
            "📶 Select Transmission SNR (0 to 20 dB):",
            min_value=0,
            max_value=20,
            value=init_snr,
            step=1,
            key="app_p5_snr_slider_select"
        )
        st.session_state["custom_snr_db"] = selected_snr_db
        st.session_state["selected_snr_db"] = selected_snr_db

    st.markdown(
        f"""
        <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 12px; padding: 18px; margin-bottom: 20px;">
            <div style="display: grid; grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr; align-items: center; text-align: center; gap: 8px;">
                <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 12px;">
                    <span style="color: #94A3B8; font-size: 11px; font-weight: 600;">SOURCE SERVER</span>
                    <div style="color: #0EA5E9; font-size: 16px; font-weight: 700; margin-top: 4px;">🖥️ {servers_str}</div>
                </div>
                <div style="color: #64748B; font-size: 20px; font-weight: 700;">↓</div>
                <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 12px;">
                    <span style="color: #94A3B8; font-size: 11px; font-weight: 600;">SEMANTIC MEANING</span>
                    <div style="color: #38BDF8; font-size: 16px; font-weight: 700; margin-top: 4px;">📦 {sem_meaning}</div>
                </div>
                <div style="color: #64748B; font-size: 20px; font-weight: 700;">↓</div>
                <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 12px;">
                    <span style="color: #94A3B8; font-size: 11px; font-weight: 600;">6G CHANNEL & SNR</span>
                    <div style="color: #10B981; font-size: 15px; font-weight: 700; margin-top: 4px;">🌐 {channel_type} ({selected_snr_db} dB)</div>
                </div>
                <div style="color: #64748B; font-size: 20px; font-weight: 700;">↓</div>
                <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 12px;">
                    <span style="color: #94A3B8; font-size: 11px; font-weight: 600;">DESTINATION NODE</span>
                    <div style="color: #F59E0B; font-size: 16px; font-weight: 700; margin-top: 4px;">📱 {nodes_str}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 10px; padding: 16px; margin-bottom: 20px;">
            <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; font-size: 13px;">
                <div><span style="color:#94A3B8;">Source Server:</span><br><b style="color:#F8FAFC; font-size: 15px;">{servers_str}</b></div>
                <div><span style="color:#94A3B8;">Destination Node:</span><br><b style="color:#38BDF8; font-size: 15px;">{nodes_str}</b></div>
                <div><span style="color:#94A3B8;">Selected SNR:</span><br><b style="color:#10B981; font-size: 15px;">{selected_snr_db:.1f} dB</b></div>
                <div><span style="color:#94A3B8;">Latency:</span><br><b style="color:#F8FAFC; font-size: 15px;">{primary_lat:.1f} ms</b></div>
                <div><span style="color:#94A3B8;">Bandwidth:</span><br><b style="color:#F8FAFC; font-size: 15px;">{primary_bw:.1f} Mbps</b></div>
                <div><span style="color:#94A3B8;">Packet Loss:</span><br><b style="color:#F8FAFC; font-size: 15px;">{primary_loss:.2f}%</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🚀 START TRANSMISSION", key="app_p5_btn_start_transmission", use_container_width=True, type="primary"):
        with st.spinner(f"🚀 Transmitting semantic image vector over {channel_type} channel at {selected_snr_db} dB SNR to destination node(s): {nodes_str}..."):
            import time
            time.sleep(0.1)
            multi_results = []
            for target_node in selected_target_nodes:
                t_info = dict(dt_engine.state.node_telemetry.get(target_node, {}))
                t_info["snr_db"] = float(selected_snr_db)
                t_info["selected_snr_db"] = float(selected_snr_db)
                
                res_node = pipeline.transmit_real_jscc(
                    pil_or_np_img=curr_img_comm,
                    destination_node_id=target_node,
                    node_telemetry=t_info,
                    dt_engine=dt_engine,
                    semantic_meaning=sem_meaning,
                    channel_type=channel_type,
                    snr_db=selected_snr_db
                )
                multi_results.append(res_node)
            
            st.session_state["multi_node_results"] = multi_results
            img_class_id = str(comm_class).lower()
            multi_results[0]["image_id"] = img_class_id
            st.session_state["last_jscc_result"] = multi_results[0]
            if str(channel_type).upper().startswith("RAYLEIGH"):
                st.session_state["results_rayleigh"] = multi_results[0]
            else:
                st.session_state["results_awgn"] = multi_results[0]

            if "tested_measurements" not in st.session_state:
                st.session_state["tested_measurements"] = []

            m_rec = {
                "image_id": img_class_id,
                "snr": int(selected_snr_db),
                "channel": channel_type,
                "mse": float(multi_results[0]["mse"]),
                "psnr": float(multi_results[0]["psnr_db"]),
                "node": primary_node
            }

            updated = False
            for idx, item in enumerate(st.session_state["tested_measurements"]):
                if item.get("image_id", img_class_id) == img_class_id and item["channel"] == channel_type and item["snr"] == int(selected_snr_db):
                    st.session_state["tested_measurements"][idx] = m_rec
                    updated = True
                    break
            if not updated:
                st.session_state["tested_measurements"].append(m_rec)
            st.rerun()

    if "multi_node_results" in st.session_state and st.session_state["multi_node_results"]:
        res_list = st.session_state["multi_node_results"]
        res_primary = res_list[0]
        
        st.markdown("---")
        st.markdown("### 📊 PACKET STATISTICS")
        
        total_sent = sum(r.get("packets_sent", 100) for r in res_list)
        total_received = sum(r.get("packets_received", 100) for r in res_list)
        total_lost = sum(r.get("packets_lost", 0) for r in res_list)
        avg_loss_pct = (total_lost / total_sent * 100.0) if total_sent > 0 else 0.0
        
        st.markdown(
            f"""
            <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 10px; padding: 18px; margin-bottom: 20px;">
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; text-align: center;">
                    <div style="background: #1E293B; border-radius: 8px; padding: 12px;">
                        <span style="color: #94A3B8; font-size: 12px; font-weight: 600;">PACKETS SENT</span>
                        <div style="color: #F8FAFC; font-size: 22px; font-weight: 700; margin-top: 4px;">{total_sent}</div>
                    </div>
                    <div style="background: #1E293B; border-radius: 8px; padding: 12px;">
                        <span style="color: #94A3B8; font-size: 12px; font-weight: 600;">PACKETS RECEIVED</span>
                        <div style="color: #10B981; font-size: 22px; font-weight: 700; margin-top: 4px;">{total_received}</div>
                    </div>
                    <div style="background: #1E293B; border-radius: 8px; padding: 12px;">
                        <span style="color: #94A3B8; font-size: 12px; font-weight: 600;">PACKETS LOST</span>
                        <div style="color: #EF4444; font-size: 22px; font-weight: 700; margin-top: 4px;">{total_lost}</div>
                    </div>
                    <div style="background: #1E293B; border-radius: 8px; padding: 12px;">
                        <span style="color: #94A3B8; font-size: 12px; font-weight: 600;">PACKET LOSS %</span>
                        <div style="color: #F59E0B; font-size: 22px; font-weight: 700; margin-top: 4px;">{avg_loss_pct:.2f}%</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(f"### 🖼️ IMAGE TRANSMISSION COMPARISON ({channel_type.upper()} CHANNEL)")
        st.caption(f"Channel: **{channel_type}** | Transmission SNR: **{float(res_primary.get('snr_db', selected_snr_db)):.1f} dB** | Reconstructed MSE: **{res_primary.get('mse', 0.0):.6f}** | Reconstructed PSNR: **{res_primary.get('psnr_db', 0.0):.2f} dB**")
        col_orig, col_recon = st.columns(2)
        
        with col_orig:
            st.image(curr_img_comm, caption=f"Original Input Image ({comm_class.upper()})", use_container_width=True)
            
        with col_recon:
            if len(res_list) == 1:
                st.image(res_primary["reconstructed_np"], caption=f"Reconstructed Image ({channel_type} Channel) | MSE: {res_primary.get('mse', 0.0):.5f} | PSNR: {res_primary.get('psnr_db', 0.0):.1f} dB", use_container_width=True)
            else:
                st.markdown(f"**Reconstructed Images across target nodes ({len(res_list)}):**")
                cols_grid = st.columns(min(len(res_list), 4))
                for idx, r in enumerate(res_list):
                    with cols_grid[idx % 4]:
                        if r.get("active", True) and "reconstructed_np" in r:
                            st.image(r["reconstructed_np"], caption=f"Node {r['destination_node']}\nMSE: {r.get('mse', 0.0):.4f}\nPSNR: {r.get('psnr_db', 0.0):.1f} dB", use_container_width=True)
                        else:
                            st.error(f"Node {r['destination_node']}\nINACTIVE")

    st.markdown("---")
    if st.button("Continue → Digital Twin", key="btn_continue_to_dt_app", use_container_width=True, type="primary"):
        navigate_to("Digital Twin")


# ====================================================================
# PAGE 5: DIGITAL TWIN
# ====================================================================
def render_digital_twin():
    render_dashboard_header(current_stage="Digital Twin")

    from digital_twin.twin import DigitalTwinEngine
    from dashboard.dt_3d import render_3d_digital_twin

    if "dt_engine" not in st.session_state or not hasattr(st.session_state["dt_engine"], "update_telemetry"):
        st.session_state["dt_engine"] = DigitalTwinEngine()

    dt_engine = st.session_state["dt_engine"]
    node_telemetry = dt_engine.state.node_telemetry

    if "selected_node" not in st.session_state or not st.session_state["selected_node"]:
        st.session_state["selected_node"] = "N3"

    selected_dest_nodes = st.session_state.get("selected_nodes", ["N1", "N3", "N7"])

    st.markdown("### 🪞 6G Network Digital Twin & Virtual Mirror")
    st.caption("Live 3D spatial simulation and real-time telemetry mirror (Server 1: N1–N5 | Server 2: N6–N10)")

    st.markdown("#### 🧊 3D Interactive Network Topology Twin")
    render_3d_digital_twin(node_telemetry, selected_dest_nodes, height=450)

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

    st.markdown("---")
    st.markdown("### 📋 SELECTED NODE TELEMETRY")

    node_options = [f"N{i}" for i in range(1, 11)]
    curr_inspect = st.session_state.get("selected_node", "N3")
    inspect_idx = node_options.index(curr_inspect) if curr_inspect in node_options else 2

    selected_inspect_node = st.selectbox(
        "Select Node to Inspect Telemetry:",
        node_options,
        index=inspect_idx,
        key="app_p5_inspect_node_select"
    )

    st.session_state["selected_node"] = selected_inspect_node

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
            key="app_p5_telemetry_scenario"
        )

    with col_act:
        st.write("")
        st.write("")
        if st.button("🔄 Sync / Update Telemetry", key="app_page5_update_telemetry_btn", use_container_width=True, type="primary"):
            dt_engine.update_telemetry(scenario=scen_choice)
            st.rerun()

    st.markdown("---")
    if st.button("Continue → Prediction", key="btn_continue_to_pred_app", use_container_width=True, type="primary"):
        navigate_to("Prediction")


# ====================================================================
# PAGE 6: PREDICTION
# ====================================================================
def render_prediction():
    render_dashboard_header(current_stage="Prediction")

    from digital_twin.twin import DigitalTwinEngine

    if "dt_engine" not in st.session_state:
        st.session_state["dt_engine"] = DigitalTwinEngine()

    dt_engine = st.session_state["dt_engine"]
    predictor = get_digital_twin_predictor()

    st.markdown("### 🔮 6G Wireless Channel AI Prediction & Forecast")
    st.caption("PyTorch LSTM Neural Network forecasting future wireless channel conditions and SNR trajectories")

    node_options = [f"N{i}" for i in range(1, 11)]
    curr_selected = st.session_state.get("selected_node", "N1")
    node_idx = node_options.index(curr_selected) if curr_selected in node_options else 0

    selected_node = st.selectbox(
        "Select Destination Node to Inspect LSTM Forecast:",
        node_options,
        index=node_idx,
        key="app_p6_node_select"
    )
    st.session_state["selected_node"] = selected_node

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
    st.markdown("#### 📈 SNR HISTORY → FUTURE SNR")
    st.caption(f"Historical channel trajectory vs. PyTorch LSTM predicted future point for node `{selected_node}`")

    hist_steps = [f"T-{len(hist_snr)-1-i}" for i in range(len(hist_snr))]
    future_step = "T+2s (Forecast)"

    fig_snr = go.Figure()

    fig_snr.add_trace(go.Scatter(
        x=hist_steps,
        y=hist_snr,
        mode="lines+markers",
        name="Historical SNR",
        line=dict(color="#38BDF8", width=3),
        marker=dict(size=8, color="#0EA5E9", symbol="circle")
    ))

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

    fig_snr.add_vline(
        x=len(hist_steps) - 1,
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
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("🔮 Refresh LSTM Forecast", key="app_p6_refresh_lstm_btn", use_container_width=True, type="primary"):
            st.rerun()

    with col_p2:
        if st.button("Continue → Adaptive Control", key="btn_continue_to_ctrl_app", use_container_width=True, type="primary"):
            navigate_to("Adaptive Control")


# ====================================================================
# PAGE 7: ADAPTIVE CONTROL
# ====================================================================
def render_adaptive_control():
    render_dashboard_header(current_stage="Adaptive Control")

    from digital_twin.twin import DigitalTwinEngine

    if "dt_engine" not in st.session_state:
        st.session_state["dt_engine"] = DigitalTwinEngine()

    dt_engine = st.session_state["dt_engine"]
    predictor = get_digital_twin_predictor()
    controller = get_adaptive_controller()

    st.markdown("### 🎛️ Proactive 6G Adaptive Control Pipeline")
    st.caption("Closed-loop semantic communication profile adaptation based on Digital Twin PyTorch LSTM channel forecasting")

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

    node_options = [f"N{i}" for i in range(1, 11)]
    curr_selected = st.session_state.get("selected_node", "N1")
    node_idx = node_options.index(curr_selected) if curr_selected in node_options else 0

    selected_node = st.selectbox(
        "Select Target Edge Node for Adaptive Control Evaluation:",
        node_options,
        index=node_idx,
        key="app_p7_node_select"
    )
    st.session_state["selected_node"] = selected_node

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
    col_ac1, col_ac2 = st.columns(2)
    with col_ac1:
        if st.button("🎛️ Re-Evaluate Proactive Adaptive Control", key="app_p7_eval_ctrl_btn", use_container_width=True, type="primary"):
            st.rerun()

    with col_ac2:
        if st.button("Continue → Results", key="btn_continue_to_results_app", use_container_width=True, type="primary"):
            navigate_to("Results")


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


# ====================================================================
# PAGE 8: RESULTS
# ====================================================================
def render_results():
    render_dashboard_header(current_stage="Results")

    from digital_twin.twin import DigitalTwinEngine

    if "dt_engine" not in st.session_state:
        st.session_state["dt_engine"] = DigitalTwinEngine()

    dt_engine = st.session_state["dt_engine"]
    pipeline = get_transmission_pipeline()

    curr_img = st.session_state["selected_image"]
    comm_class = st.session_state.get("actual_class", "dog").upper()
    img_class_id = comm_class.lower()
    sem_meaning = st.session_state.get("semantic_meaning", comm_class).upper()
    selected_node = st.session_state.get("selected_node", "N3")
    curr_channel = st.session_state.get("comm_channel_type", "AWGN")
    curr_snr = float(st.session_state.get("selected_snr_db", 15.0))

    source_server = pipeline.determine_source_server(selected_node)
    target_telemetry = dict(dt_engine.state.node_telemetry.get(selected_node, {
        "snr_db": curr_snr, "latency_ms": 10.0, "bandwidth_mbps": 100.0, "packet_loss_pct": 0.1, "status": "ACTIVE"
    }))
    target_telemetry["snr_db"] = curr_snr

    last_res = st.session_state.get("last_jscc_result")
    is_match = (
        last_res is not None
        and str(last_res.get("image_id", img_class_id)).lower() == img_class_id
        and str(last_res.get("channel_type", "AWGN")).upper() == str(curr_channel).upper()
        and abs(float(last_res.get("snr_db", -1.0)) - curr_snr) < 1e-3
    )

    if is_match:
        res_j = last_res
    else:
        res_j = pipeline.transmit_real_jscc(
            pil_or_np_img=curr_img,
            destination_node_id=selected_node,
            node_telemetry=target_telemetry,
            dt_engine=dt_engine,
            semantic_meaning=sem_meaning,
            channel_type=curr_channel,
            snr_db=curr_snr
        )
        res_j["image_id"] = img_class_id
        st.session_state["last_jscc_result"] = res_j
        if str(curr_channel).upper().startswith("RAYLEIGH"):
            st.session_state["results_rayleigh"] = res_j
        else:
            st.session_state["results_awgn"] = res_j

        if "tested_measurements" not in st.session_state:
            st.session_state["tested_measurements"] = []

        m_rec = {
            "image_id": img_class_id,
            "snr": int(curr_snr),
            "channel": curr_channel,
            "mse": float(res_j["mse"]),
            "psnr": float(res_j["psnr_db"]),
            "node": selected_node
        }
        updated = False
        for idx, item in enumerate(st.session_state["tested_measurements"]):
            if item.get("image_id", img_class_id) == img_class_id and item["channel"] == curr_channel and item["snr"] == int(curr_snr):
                st.session_state["tested_measurements"][idx] = m_rec
                updated = True
                break
        if not updated:
            st.session_state["tested_measurements"].append(m_rec)

    res_awgn = st.session_state.get("results_awgn")
    res_rayleigh = st.session_state.get("results_rayleigh")
    has_both = (
        res_awgn is not None and str(res_awgn.get("image_id", img_class_id)).lower() == img_class_id
        and res_rayleigh is not None and str(res_rayleigh.get("image_id", img_class_id)).lower() == img_class_id
    )

    st.markdown("### 📊 Experiment Results & Semantic Transmission Evaluation")
    st.caption("End-to-End Deep JSCC Transmission and Digital Twin Telemetry Metrics")

    if has_both:
        st.markdown("---")
        st.markdown("#### ⚔️ DEEP JSCC CHANNEL MODEL COMPARISON (AWGN vs RAYLEIGH FADING)")
        st.caption("Actual empirical reconstruction metrics measured across both channel models for the same image and telemetry SNR.")
        col_awgn_ui, col_ray_ui = st.columns(2)
        with col_awgn_ui:
            st.markdown("##### 🌐 AWGN CHANNEL")
            st.image(res_awgn["reconstructed_np"], caption=f"AWGN Reconstructed Image — {img_class_id}", use_container_width=True)
            st.metric("AWGN MSE", f"{res_awgn.get('mse', 0.0):.6f}")
            st.metric("AWGN PSNR", f"{res_awgn.get('psnr_db', 0.0):.2f} dB")
        with col_ray_ui:
            st.markdown("##### 🌐 RAYLEIGH FADING CHANNEL")
            st.image(res_rayleigh["reconstructed_np"], caption=f"Rayleigh Reconstructed Image — {img_class_id}", use_container_width=True)
            st.metric("Rayleigh MSE", f"{res_rayleigh.get('mse', 0.0):.6f}")
            st.metric("Rayleigh PSNR", f"{res_rayleigh.get('psnr_db', 0.0):.2f} dB")
    else:
        st.info("ℹ️ **Run both channels to compare**")
        st.caption("Currently showing single channel run result. To compare AWGN vs Rayleigh Fading side-by-side, return to the Communication page, select the other channel model, and click Start Transmission.")

    st.markdown("---")
    st.markdown("#### 🖼️ IMAGE COMPARISON")

    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.markdown("**ORIGINAL IMAGE**")
        st.image(curr_img, caption=f"Original Input Image ({comm_class})", use_container_width=True)

    with col_img2:
        st.markdown("**RECONSTRUCTED IMAGE**")
        rec_disp = res_j.get("reconstructed_np")
        if rec_disp is None or not isinstance(rec_disp, np.ndarray) or rec_disp.ndim < 2:
            if hasattr(curr_img, "convert"):
                rec_disp = np.array(curr_img.convert("RGB"), dtype=np.float32) / 255.0
            else:
                rec_disp = np.array(curr_img, dtype=np.float32)
        channel_label_str = "Rayleigh Fading" if "Rayleigh" in str(res_j.get('channel_type', curr_channel)) else "AWGN"
        st.image(rec_disp, caption=f"Reconstructed Image — {img_class_id} | {channel_label_str}", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📈 PERFORMANCE METRICS")

    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    col_m1.metric("MSE", f"{res_j.get('mse', 0.0):.6f}")
    col_m2.metric("PSNR", f"{res_j.get('psnr_db', 0.0):.2f} dB")
    col_m3.metric("SNR", f"{res_j.get('snr_db', 0.0):.2f} dB")
    col_m4.metric("PACKET LOSS", f"{res_j.get('packet_loss_pct', 0.0):.2f}%")
    col_m5.metric("LATENCY", f"{res_j.get('latency_ms', 0.0):.2f} ms")
    col_m6.metric("BANDWIDTH", f"{res_j.get('bandwidth_mbps', 0.0):.2f} Mbps")



    # --- NEW SECTION: SNR VS. PSNR PERFORMANCE EVALUATION SWEEP ---
    st.markdown("---")
    st.markdown("#### 📈 SNR vs. PSNR MEASUREMENT GRAPH")
    st.caption(f"Plotting ONLY actual user-tested transmissions for image `{comm_class}` across SNR values (0 to 20 dB). Select SNR values on the Communication page and run transmission to add points.")

    all_user_pts = st.session_state.get("tested_measurements", [])
    user_pts = [p for p in all_user_pts if p.get("image_id", img_class_id).lower() == img_class_id]
    
    awgn_user_pts = sorted([p for p in user_pts if p["channel"] == "AWGN"], key=lambda x: x["snr"])
    ray_user_pts = sorted([p for p in user_pts if p["channel"] == "Rayleigh"], key=lambda x: x["snr"])

    fig_interactive = go.Figure()

    if awgn_user_pts:
        fig_interactive.add_trace(go.Scatter(
            x=[p["snr"] for p in awgn_user_pts],
            y=[p["psnr"] for p in awgn_user_pts],
            mode="lines+markers",
            name="AWGN Channel (Measured)",
            line=dict(color="#38BDF8", width=3),
            marker=dict(size=9, color="#0EA5E9", symbol="circle")
        ))

    if ray_user_pts:
        fig_interactive.add_trace(go.Scatter(
            x=[p["snr"] for p in ray_user_pts],
            y=[p["psnr"] for p in ray_user_pts],
            mode="lines+markers",
            name="Rayleigh Fading Channel (Measured)",
            line=dict(color="#F59E0B", width=3, dash="dash"),
            marker=dict(size=9, color="#D97706", symbol="diamond")
        ))

    fig_interactive.update_layout(
        template="plotly_dark",
        title=f"User-Tested SNR vs. PSNR Measurements for '{comm_class}' ({len(user_pts)} Total Measured Points)",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        xaxis=dict(title="Signal-to-Noise Ratio (SNR in dB)", dtick=1, range=[-0.5, 20.5], showgrid=True, gridcolor="#1E293B"),
        yaxis=dict(title="Peak Signal-to-Noise Ratio (PSNR in dB)", showgrid=True, gridcolor="#1E293B"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    if user_pts:
        st.plotly_chart(fig_interactive, use_container_width=True)
        with st.expander("📋 View Measured Data Table", expanded=False):
            df_user = pd.DataFrame(user_pts)
            st.dataframe(df_user, use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ **No user-tested SNR measurements recorded yet.** Select an SNR value (0 to 20 dB) and Channel Model on the Communication page and click Start Transmission to plot actual measured points.")

    with st.expander("📊 View Complete 42-Point Baseline SNR Sweep Benchmark (0 to 20 dB)", expanded=False):
        sweep_data = compute_snr_vs_psnr_sweep(curr_img)
        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(
            x=sweep_data["snr_range"], y=sweep_data["awgn_psnr"],
            mode="lines+markers", name="AWGN Channel (Baseline Sweep)",
            line=dict(color="#38BDF8", width=3), marker=dict(size=6, color="#0EA5E9")
        ))
        fig_sweep.add_trace(go.Scatter(
            x=sweep_data["snr_range"], y=sweep_data["rayleigh_psnr"],
            mode="lines+markers", name="Rayleigh Fading (Baseline Sweep)",
            line=dict(color="#F59E0B", width=3, dash="dash"), marker=dict(size=6, color="#D97706")
        ))
        fig_sweep.update_layout(
            template="plotly_dark",
            title="Complete 42-Point SNR vs. PSNR Baseline Curve",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            xaxis=dict(title="SNR (dB)", dtick=1, range=[-0.5, 20.5]),
            yaxis=dict(title="PSNR (dB)")
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

    st.markdown("---")
    if st.button("🔄 Restart Experiment Pipeline ➔ Overview", key="btn_restart_exp_app", use_container_width=True, type="primary"):
        navigate_to("Overview")


# ====================================================================
# PAGE 9: ABOUT
# ====================================================================
def render_about():
    render_dashboard_header(current_stage="About")

    st.markdown("### 🎯 Project Objective")
    st.write(
        "The primary objective of this research project is to demonstrate end-to-end semantic communication "
        "in next-generation 6G wireless networks. By deploying a Digital Twin framework, real-time network "
        "telemetry (SNR, latency, bandwidth, packet loss) is mirrored to enable proactive PyTorch LSTM channel prediction "
        "and closed-loop adaptive control over PyTorch Deep Joint Source-Channel Coding (JSCC) transmission."
    )

    st.markdown("---")

    st.markdown("### 🖼️ Dataset Benchmark")
    st.write(
        "**CIFAR-10 Benchmark Dataset:** 60,000 32×32×3 RGB images across 10 categories (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)."
    )

    st.markdown("---")

    st.markdown("### 🧠 Deep Learning Architectures")
    st.write(
        "1. **ResNet-18 Semantic Encoder:** Pretrained CNN deriving 512-D pooling vector representation.\n"
        "2. **Deep JSCC Codec:** Convolutional Encoder + Differentiable 6G Wireless Noise Channels (AWGN & Rayleigh Fading) + Transposed Convolutional Decoder.\n"
        "3. **PyTorch LSTM Channel Predictor:** 2-layer LSTM forecasting future SNR trajectories 2s ahead."
    )

    st.markdown("---")

    st.markdown("### 🌐 Hierarchical Network Architecture")
    st.write(
        "2 Edge Servers (`Server 1` & `Server 2`) routing to 10 Edge Nodes (`N1` to `N10`) with live telemetry feedback."
    )


# ====================================================================
# MAIN APPLICATION ROUTER (STATE-BASED SINGLE PAGE SWITCHING)
# ====================================================================
if current_page == "Overview":
    render_overview()
elif current_page == "Image & Semantics":
    render_image_semantics()
elif current_page == "Network":
    render_network()
elif current_page == "Communication":
    render_communication()
elif current_page == "Digital Twin":
    render_digital_twin()
elif current_page == "Prediction":
    render_prediction()
elif current_page == "Adaptive Control":
    render_adaptive_control()
elif current_page == "Results":
    render_results()
elif current_page == "About":
    render_about()

# Render Global Dark Tech Footer
render_footer()
