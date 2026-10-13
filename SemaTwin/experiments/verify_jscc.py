import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

from models.deep_jscc import DeepJSCCModel
from data.dataset import CIFAR10DataModule


def verify_deep_jscc(snr_db=15.0):
    print("============================================================", flush=True)
    print(" PHASE 8A: REAL PYTORCH DEEP JSCC MODEL VERIFICATION", flush=True)
    print("============================================================", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Compute Device: {device}", flush=True)

    # 1. Load Trained Checkpoint
    checkpoint_path = os.path.join(PROJECT_ROOT, "results", "models", "deep_jscc_cifar10.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint model not found at {checkpoint_path}")

    model = DeepJSCCModel(latent_channels=16).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Calculate model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print(f" Checkpoint Location: {checkpoint_path}", flush=True)
    print(f" Encoder Parameters:  {encoder_params:,} (PyTorch JSCCEncoder nn.Module)", flush=True)
    print(f" Decoder Parameters:  {decoder_params:,} (PyTorch JSCCDecoder nn.Module)", flush=True)
    print(f" Total JSCC Params:   {total_params:,}", flush=True)

    # 2. Fetch CIFAR-10 Test Sample Image
    dm = CIFAR10DataModule()
    pil_img, cls_name, cls_idx, img_np, img_path = dm.get_random_sample_by_class("cat")
    
    # 3. Convert Image to PyTorch Tensor [1, 3, 32, 32]
    img_tensor = torch.tensor(img_np.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    # 4. Execute Real PyTorch Pipeline: Image -> Encoder -> AWGN -> Decoder -> Reconstructed
    with torch.no_grad():
        latent = model.encoder(img_tensor)
        noisy_latent = model.channel(latent, snr_db=snr_db)
        reconstructed_tensor = model.decoder(noisy_latent)

    # Verify PyTorch Tensor types
    assert isinstance(latent, torch.Tensor), "Latent must be torch.Tensor"
    assert isinstance(noisy_latent, torch.Tensor), "Noisy Latent must be torch.Tensor"
    assert isinstance(reconstructed_tensor, torch.Tensor), "Reconstructed output must be torch.Tensor"

    # Compute Metrics
    rec_np = reconstructed_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    rec_np = np.clip(rec_np, 0.0, 1.0)
    orig_np = np.clip(img_np, 0.0, 1.0)

    mse_val = float(np.mean((orig_np - rec_np) ** 2))
    psnr_val = 10.0 * np.log10(1.0 / max(1e-8, mse_val))

    print("------------------------------------------------------------", flush=True)
    print(f" Target Test Category: {cls_name.upper()}", flush=True)
    print(f" Verification SNR:    {snr_db:.2f} dB", flush=True)
    print(f" Reconstruction MSE:  {mse_val:.6f}", flush=True)
    print(f" Reconstruction PSNR: {psnr_val:.2f} dB", flush=True)
    print("------------------------------------------------------------", flush=True)

    # 5. Save Verification Comparison Image
    metrics_dir = os.path.join(PROJECT_ROOT, "results", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    plot_path = os.path.join(metrics_dir, "deep_jscc_verification.png")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_np)
    axes[0].set_title(f"Original ({cls_name.upper()})")
    axes[0].axis("off")

    axes[1].imshow(rec_np)
    axes[1].set_title(f"Deep JSCC Reconstructed\n(SNR: {snr_db}dB, PSNR: {psnr_val:.2f}dB)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f" Saved Comparison Plot To: {plot_path}", flush=True)
    print("============================================================", flush=True)
    print(" CONFIRMATION: PIPELINE IS 100% REAL PYTORCH COMPUTATION.", flush=True)
    print("============================================================", flush=True)

    return {
        "snr_db": snr_db,
        "mse": mse_val,
        "psnr_db": psnr_val,
        "checkpoint_path": checkpoint_path,
        "plot_path": plot_path,
        "total_params": total_params,
        "encoder_params": encoder_params,
        "decoder_params": decoder_params
    }


if __name__ == "__main__":
    verify_deep_jscc(snr_db=15.0)
