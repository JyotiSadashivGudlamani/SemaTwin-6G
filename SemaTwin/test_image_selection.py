import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import CIFAR10DataModule, CIFAR10_CLASSES
from models.resnet_semantic import ResNet18CIFAR10
import torch


def test_correct_image_selection():
    dm = CIFAR10DataModule()
    model = ResNet18CIFAR10(num_classes=10, pretrained=True)
    pth_path = os.path.join(PROJECT_ROOT, "results", "models", "resnet18_cifar10.pth")

    if os.path.exists(pth_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()

    print("==================================================")
    print("TEST: CORRECTLY CLASSIFIED IMAGE SELECTION")
    print("==================================================")

    all_matched = True
    for cls in CIFAR10_CLASSES:
        pil_img, cls_name, cls_idx, img_np, chosen_p = dm.get_correctly_classified_sample_by_class(cls, model=model)
        pred_res = model.predict_image(pil_img)
        pred_cls = pred_res["predicted_class"].lower()
        conf = pred_res["confidence_pct"]

        matched = (cls_name.lower() == pred_cls)
        if not matched:
            all_matched = False

        status_str = "MATCH" if matched else "MISMATCH"
        print(f"Class: {cls:<10} | Actual: {cls_name:<10} | Predicted: {pred_cls:<10} | Conf: {conf:6.2f}% | Status: {status_str}")

    print("==================================================")
    print(f"All 10 Classes Verified Matched: {'YES' if all_matched else 'NO'}")
    print("==================================================")


if __name__ == "__main__":
    test_correct_image_selection()
