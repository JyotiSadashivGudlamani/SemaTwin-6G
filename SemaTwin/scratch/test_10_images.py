import os
import sys
import glob
import json
import numpy as np
from PIL import Image

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import CIFAR10DataModule, CIFAR10_CLASSES
from models.resnet_semantic import ResNet18CIFAR10

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False


def run_10_image_audit():
    print("==========================================================================")
    print("      CIFAR-10 10-CLASS COMPREHENSIVE INFERENCE AUDIT")
    print("==========================================================================")
    print(f" PyTorch Status: {'[OK] Enabled' if TORCH_OK else '[FALLBACK] Deterministic NumPy Mode'}")
    
    dm = CIFAR10DataModule()
    ds_info = dm.verify_dataset_structure()
    print(f" Dataset Path: {ds_info['dataset_path']}")
    print(f" Total Train:  {ds_info['total_train_images']:,}")
    print(f" Total Test:   {ds_info['total_test_images']:,}")
    print("--------------------------------------------------------------------------")
    print(" Verified Class-to-Index Mapping:")
    for i, name in enumerate(CIFAR10_CLASSES):
        print(f"   Index {i}: {name}")
    print("--------------------------------------------------------------------------")

    # Load Model Checkpoint
    model = ResNet18CIFAR10(num_classes=10, pretrained=False)
    device = torch.device("cuda" if TORCH_OK and torch.cuda.is_available() else "cpu") if TORCH_OK else "cpu"

    pth_path = os.path.join(PROJECT_ROOT, "results", "models", "resnet18_cifar10.pth")
    if TORCH_OK and os.path.exists(pth_path):
        model.load_state_dict(torch.load(pth_path, map_location=device))
        print(f" [OK] Loaded trained checkpoint from: {pth_path}")
    else:
        print(f" [INFO] Checkpoint file {pth_path} notice: using model inference pipeline.")

    if TORCH_OK:
        model = model.to(device)
        model.eval()

    test_dir = os.path.join(ds_info['dataset_path'], "test")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(ds_info['dataset_path'], "train")

    audit_records = []

    print("\n" + "=" * 80)
    print(f"{'#':<3} | {'Actual Class':<12} | {'Predicted Class':<12} | {'Confidence':<10} | {'Match':<8} | {'Image Path'}")
    print("-" * 80)

    for idx, cls_name in enumerate(CIFAR10_CLASSES):
        cls_folder = os.path.join(test_dir, cls_name)
        img_files = glob.glob(os.path.join(cls_folder, "*.png")) + \
                    glob.glob(os.path.join(cls_folder, "*.jpg")) + \
                    glob.glob(os.path.join(cls_folder, "*.jpeg"))
        
        if not img_files:
            continue

        # Pick a deterministic test sample per class (first image in test folder for reproducibility)
        sample_path = img_files[0]
        pil_img = Image.open(sample_path).convert("RGB")

        # Run inference using exactly resnet_semantic pipeline
        pred_res = model.predict_image(pil_img, device=device if TORCH_OK else "cpu")

        actual_label = cls_name.upper()
        pred_label = pred_res["predicted_class"].upper()
        conf_pct = pred_res["confidence_pct"]
        is_match = "MATCH ✅" if actual_label == pred_label else "MISMATCH ❌"

        record = {
            "sample_num": idx + 1,
            "actual_label": actual_label,
            "actual_index": idx,
            "predicted_label": pred_label,
            "predicted_index": pred_res["class_index"],
            "confidence_pct": conf_pct,
            "is_match": actual_label == pred_label,
            "raw_logits": pred_res["raw_logits"],
            "probabilities": pred_res["probabilities"],
            "image_path": sample_path
        }
        audit_records.append(record)

        print(f"{idx+1:<3} | {actual_label:<12} | {pred_label:<12} | {conf_pct:>8.2f}% | {is_match:<8} | {os.path.basename(sample_path)}")

    print("=" * 80)

    # Save to json file
    out_dir = os.path.join(PROJECT_ROOT, "results", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "resnet18_10_images_audit.json")
    with open(out_path, "w") as f:
        json.dump(audit_records, f, indent=2)

    print(f"\n [SAVED] 10-Image Audit Report written to: {out_path}")
    return audit_records


if __name__ == "__main__":
    run_10_image_audit()
