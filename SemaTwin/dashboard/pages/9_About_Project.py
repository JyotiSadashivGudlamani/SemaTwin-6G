import streamlit as st
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.ui_components import render_dashboard_header
from dashboard.styles import render_footer

st.set_page_config(page_title="9. ℹ️ About", page_icon="ℹ️", layout="wide")

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
    "**CIFAR-10 Benchmark Dataset:**\n"
    "- **Image Dimensions:** 32 × 32 × 3 RGB color images.\n"
    "- **Total Images:** 60,000 images divided into 50,000 training and 10,000 test samples.\n"
    "- **Categories (10 classes):** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck."
)

st.markdown("---")

st.markdown("### 🧠 Deep Learning Models")
st.write(
    "1. **Semantic Understanding Model (ResNet-18):**\n"
    "   - Pretrained ResNet-18 convolutional neural network (`models/resnet_semantic.py`).\n"
    "   - Extracts a **512-dimensional semantic feature vector** from the average pooling layer (`avgpool`) before final classification.\n"
    "2. **Deep Joint Source-Channel Coding Model (Deep JSCC):**\n"
    "   - **`JSCCEncoder`**: Convolutional network mapping original image into compact latent semantic channels (`models/deep_jscc.py`).\n"
    "   - **`AWGNChannel` / `RayleighChannel`**: Differentiable 6G wireless channels adding Gaussian noise & Rayleigh fading corresponding to live node SNR.\n"
    "   - **`JSCCDecoder`**: Transposed convolutional network reconstructing the image directly at the destination node.\n"
    "3. **Channel Prediction Model (PyTorch LSTM Predictor):**\n"
    "   - PyTorch LSTM network (`DigitalTwinLSTMPredictor` in `models/dt_lstm.py` / `digital_twin/predictor.py`).\n"
    "   - Predicts 2-second future SNR trajectories based on historical telemetry sequences."
)

st.markdown("---")

st.markdown("### 🌐 Network Architecture")
st.write(
    "- **Hierarchical 6G Edge Topology:**\n"
    "  - **Root Node:** 6G Semantic Core Router.\n"
    "  - **Edge Content Servers (2):** `Server 1` (manages Nodes N1–N5) and `Server 2` (manages Nodes N6–N10).\n"
    "  - **Destination Nodes (10):** Edge devices (`N1` to `N10`) receiving semantic data streams.\n"
    "- **Sub-THz Wireless Links:** High-frequency 6G cells with simulated path loss, latency, and packet drop mechanics."
)

st.markdown("---")

st.markdown("### 🛠️ Technologies Used")
col_t1, col_t2 = st.columns(2)
with col_t1:
    st.write(
        "- **Deep Learning Framework:** PyTorch, Torchvision\n"
        "- **Web Dashboard:** Streamlit\n"
        "- **Data Visualization:** Plotly, Three.js"
    )
with col_t2:
    st.write(
        "- **Data Processing:** NumPy, Pandas\n"
        "- **Image Processing:** PIL (Pillow)\n"
        "- **Model Checkpoints:** Pretrained PyTorch `.pth` models"
    )

st.markdown("---")

st.markdown("### 🔄 Complete End-to-End Pipeline")
st.write(
    "1. **Image Selection & Semantic Extraction:** A CIFAR-10 image is processed through ResNet-18 to derive its predicted class and 512-D semantic embedding vector.\n"
    "2. **Server Routing & Topology:** The destination node choice (N1–N10) automatically assigns the source server (Server 1 or Server 2).\n"
    "3. **Digital Twin Mirroring:** Live node status, SNR, latency, bandwidth, and real packet loss are computed and monitored.\n"
    "4. **Deep JSCC Transmission:** Image tensors pass through the PyTorch JSCC encoder, AWGN noisy channel, and JSCC decoder.\n"
    "5. **LSTM Future Channel Prediction:** Historical telemetry is fed into the PyTorch LSTM model to project future SNR 2 seconds ahead.\n"
    "6. **Proactive Adaptive Control:** The controller evaluates predicted SNR and automatically selects the optimal communication profile (High Fidelity JSCC-A vs. Robust JSCC-B)."
)

render_footer()
