import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from data.dataset import CIFAR10DataModule
from semantic.feature_extractor import SemanticFeatureExtractor
from network.topology import NetworkTopology
from digital_twin.twin import DigitalTwinEngine
from digital_twin.predictor import DigitalTwinPredictor
from control.adaptive_controller import ProactiveAdaptiveController

def main():
    print("======================================================================")
    print(" 📡 DIGITAL TWIN FOR SEMANTIC COMMUNICATION IN 6G NETWORKS")
    print("======================================================================")
    
    # 1. Dataset Module
    dm = CIFAR10DataModule()
    info = dm.verify_dataset_structure()
    print(f"[1/6 DATASET] Verified {info['total_images']} images from: {info['dataset_path']}")

    # 2. Semantic Extractor
    extractor = SemanticFeatureExtractor()
    sample_img, cls_name, _, img_np, _ = dm.get_random_sample_by_class("dog")
    res = extractor.process_image(sample_img)
    print(f"[2/6 SEMANTIC] Extracted {res['feature_dim']}-dim Vector for Class: {res['predicted_class'].upper()} ({res['confidence_pct']}%)")

    # 3. Network Topology
    topo = NetworkTopology()
    topo.update_all_telemetry()
    summary = topo.get_topology_summary()
    print(f"[3/6 NETWORK] Topology Active ({summary['total_servers']} Servers, {summary['total_nodes']} Nodes)")

    # 4. Digital Twin Engine
    twin = DigitalTwinEngine(topo)
    twin.sync()
    print("[4/6 DIGITAL TWIN] Virtual Mirror Synced & Plotly Graph Ready")

    # 5. LSTM Predictor & Adaptive Controller
    predictor = DigitalTwinPredictor()
    controller = ProactiveAdaptiveController()
    pred_res = predictor.predict_node_future_snr(twin.state, "Node 3")
    profile = controller.select_profile(pred_res["predicted_future_snr"])
    print(f"[5/6 PREDICTION] Node 3 Predicted Future SNR: {pred_res['predicted_future_snr']} dB -> Profile: {profile['profile']}")

    # 6. Streamlit Launch Command
    print("----------------------------------------------------------------------")
    print("[6/6 DASHBOARD] To launch interactive Streamlit Research Dashboard, run:")
    print("      streamlit run dashboard/app.py")
    print("======================================================================")

if __name__ == "__main__":
    main()
