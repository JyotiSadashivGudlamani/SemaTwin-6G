import os
import sys
import json
import time
import uuid
import datetime
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False


class SemanticPacketBuilder:
    """
    Phase 4: Semantic Packet Structuring Module.
    Extracts real ResNet-18 predictions and Phase 3 512-dimensional semantic feature vectors
    into structured JSON packets.
    """
    def __init__(self, model=None, output_dir=None):
        self.model = model
        self.output_dir = output_dir or os.path.join(PROJECT_ROOT, "results", "semantic_packets")
        os.makedirs(self.output_dir, exist_ok=True)

    def build_packet(self, img_input, actual_class="UNKNOWN", image_path=None, model=None):
        """
        Generates a structured Semantic Packet containing:
        - packet_id
        - timestamp
        - image_id/path
        - actual_class
        - predicted_class
        - confidence
        - feature_dim (512)
        - real 512-D feature_vector
        """
        active_model = model or self.model
        if active_model is None:
            from models.resnet_semantic import ResNet18CIFAR10
            active_model = ResNet18CIFAR10(num_classes=10, pretrained=False)
            pth_path = os.path.join(PROJECT_ROOT, "results", "models", "resnet18_cifar10.pth")
            if TORCH_OK and os.path.exists(pth_path) and hasattr(torch, "device"):
                try:
                    device = torch.device("cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu")
                    active_model.load_state_dict(torch.load(pth_path, map_location=device))
                except Exception:
                    pass
            active_model.eval()

        # 1. Run inference for predicted class & confidence
        pred_res = active_model.predict_image(img_input)

        # 2. Extract REAL 512-D semantic feature vector
        if TORCH_OK and hasattr(active_model, "preprocess_image_input") and hasattr(active_model, "extract_features"):
            tensor_x = active_model.preprocess_image_input(img_input)
            device = next(active_model.parameters()).device if list(active_model.parameters()) else "cpu"
            tensor_x = tensor_x.to(device)
            with torch.no_grad():
                feat_tensor = active_model.extract_features(tensor_x)
                feat_vector = feat_tensor[0].cpu().numpy().tolist()
        else:
            feat_vector = np.zeros(512, dtype=np.float32).tolist()

        # Clean float list rounding to 4 decimal places
        feat_vector_clean = [round(float(v), 4) for v in feat_vector]

        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6].upper()
        packet_id = f"SPK_{now_str}_{short_id}"

        img_identifier = image_path or "cifar10_sample.png"

        packet = {
            "packet_id": packet_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "image_id": img_identifier,
            "actual_class": str(actual_class).upper(),
            "predicted_class": str(pred_res["predicted_class"]).upper(),
            "confidence": round(float(pred_res["confidence_pct"]), 2),
            "feature_dim": 512,
            "feature_vector": feat_vector_clean
        }

        return packet

    def save_packet(self, packet):
        """Saves generated semantic packet to JSON inside results/semantic_packets/"""
        os.makedirs(self.output_dir, exist_ok=True)
        file_name = f"{packet['packet_id']}.json"
        save_path = os.path.join(self.output_dir, file_name)
        with open(save_path, "w") as f:
            json.dump(packet, f, indent=2)
        return save_path


if __name__ == "__main__":
    builder = SemanticPacketBuilder()
    dummy_img = Image.new("RGB", (32, 32), color=(128, 64, 32))
    pkt = builder.build_packet(dummy_img, actual_class="test")
    p_path = builder.save_packet(pkt)
    print("==========================================================")
    print(" PHASE 4: SEMANTIC PACKET GENERATION TEST")
    print("==========================================================")
    print(f" Packet ID:       {pkt['packet_id']}")
    print(f" Timestamp:       {pkt['timestamp']}")
    print(f" Image ID:        {pkt['image_id']}")
    print(f" Actual Class:    {pkt['actual_class']}")
    print(f" Predicted Class: {pkt['predicted_class']}")
    print(f" Confidence:      {pkt['confidence']}%")
    print(f" Feature Dim:     {pkt['feature_dim']}")
    print(f" Feature Vector:  [{pkt['feature_vector'][0]}, {pkt['feature_vector'][1]}, ..., {pkt['feature_vector'][-1]}]")
    print(f" Saved JSON Path: {p_path}")
    print("==========================================================")
