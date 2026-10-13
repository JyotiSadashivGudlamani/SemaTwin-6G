import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import CIFAR10DataModule
from models.resnet_semantic import ResNet18CIFAR10
from semantic.feature_extractor import ResNet18SemanticFeatureExtractor
from digital_twin.twin import DigitalTwinEngine
from communication.transmission import SemanticTransmissionPipeline
from dashboard.ui_components import render_dashboard_header

st.set_page_config(page_title="4. 📡 Communication", page_icon="📡", layout="wide")

render_dashboard_header(current_stage="Communication")

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

# Selected destination nodes from Network Topology
selected_target_nodes = st.session_state.get("selected_nodes", ["N1", "N3", "N7"])
if not selected_target_nodes:
    selected_target_nodes = ["N1"]

primary_node = selected_target_nodes[0]
st.session_state["selected_node"] = primary_node

curr_img_comm = st.session_state["selected_image"]
comm_class = st.session_state["actual_class"]
sem_meaning = st.session_state.get("semantic_meaning", comm_class.upper()).upper()

# Source Servers
servers_set = set()
for nid in selected_target_nodes:
    servers_set.add(pipeline.determine_source_server(nid))
servers_str = ", ".join(sorted(list(servers_set)))
nodes_str = ", ".join(selected_target_nodes)

# Telemetry for Primary Node
primary_t = dt_engine.state.node_telemetry.get(primary_node, {})
primary_snr = primary_t.get("snr_db", 15.0)
primary_lat = primary_t.get("latency_ms", 25.0)
primary_bw = primary_t.get("bandwidth_mbps", 100.0)
primary_loss = primary_t.get("packet_loss_pct", 0.0)

st.markdown("### 📡 6G Transmission Monitoring Interface")
st.caption("End-to-End Deep JSCC Semantic Transmission & Real-Time Monitoring")

# --- 1. VISUAL TRANSMISSION FLOW ---
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
                <span style="color: #94A3B8; font-size: 11px; font-weight: 600;">6G CHANNEL</span>
                <div style="color: #10B981; font-size: 16px; font-weight: 700; margin-top: 4px;">🌐 AWGN Channel</div>
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

# --- 2. TRANSMISSION TELEMETRY & CHANNEL CONDITIONS ---
st.markdown(
    f"""
    <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 10px; padding: 16px; margin-bottom: 20px;">
        <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; font-size: 13px;">
            <div><span style="color:#94A3B8;">Source Server:</span><br><b style="color:#F8FAFC; font-size: 15px;">{servers_str}</b></div>
            <div><span style="color:#94A3B8;">Destination Node:</span><br><b style="color:#38BDF8; font-size: 15px;">{nodes_str}</b></div>
            <div><span style="color:#94A3B8;">SNR:</span><br><b style="color:#F8FAFC; font-size: 15px;">{primary_snr:.1f} dB</b></div>
            <div><span style="color:#94A3B8;">Latency:</span><br><b style="color:#F8FAFC; font-size: 15px;">{primary_lat:.1f} ms</b></div>
            <div><span style="color:#94A3B8;">Bandwidth:</span><br><b style="color:#F8FAFC; font-size: 15px;">{primary_bw:.1f} Mbps</b></div>
            <div><span style="color:#94A3B8;">Packet Loss:</span><br><b style="color:#F8FAFC; font-size: 15px;">{primary_loss:.1f}%</b></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- 3. PROMINENT START TRANSMISSION BUTTON ---
if st.button("🚀 START TRANSMISSION", key="p5_btn_start_transmission", use_container_width=True, type="primary"):
    with st.spinner(f"🚀 Transmitting semantic image vector to destination node(s): {nodes_str}..."):
        time.sleep(0.3)
        multi_results = []
        for target_node in selected_target_nodes:
            t_info = dict(dt_engine.state.node_telemetry.get(target_node, {}))
            t_info["snr_db"] = float(st.session_state.get("selected_snr_db", 15.0))
            t_info["selected_snr_db"] = float(st.session_state.get("selected_snr_db", 15.0))
            res_node = pipeline.transmit_real_jscc(
                pil_or_np_img=curr_img_comm,
                destination_node_id=target_node,
                node_telemetry=t_info,
                dt_engine=dt_engine,
                semantic_meaning=sem_meaning
            )
            multi_results.append(res_node)
        
        st.session_state["multi_node_results"] = multi_results
        st.session_state["last_jscc_result"] = multi_results[0]

# --- 4. PACKET STATISTICS & RESULTS ---
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
    
    st.markdown("### 🖼️ IMAGE TRANSMISSION COMPARISON")
    col_orig, col_recon = st.columns(2)
    
    with col_orig:
        st.image(curr_img_comm, caption=f"Original Input Image ({comm_class.upper()})", use_container_width=True)
        
    with col_recon:
        if len(res_list) == 1:
            st.image(res_primary["reconstructed_np"], caption=f"Reconstructed Image at Destination ({res_primary['destination_node']})", use_container_width=True)
        else:
            st.markdown(f"**Reconstructed Images across target nodes ({len(res_list)}):**")
            cols_grid = st.columns(min(len(res_list), 4))
            for idx, r in enumerate(res_list):
                with cols_grid[idx % 4]:
                    if r.get("active", True) and "reconstructed_np" in r:
                        st.image(r["reconstructed_np"], caption=f"Node {r['destination_node']}\nLoss: {r['packet_loss_pct']:.1f}%", use_container_width=True)
                    else:
                        st.error(f"Node {r['destination_node']}\nINACTIVE")

st.markdown("---")

# --- NAVIGATION CONTINUATION BUTTON ---
if st.button("Continue → Digital Twin", key="btn_continue_to_dt_p4", use_container_width=True, type="primary"):
    try:
        st.switch_page("pages/5_Digital_Twin.py")
    except Exception:
        st.info("👈 Select **🪞 Digital Twin** in the sidebar navigation to proceed.")
