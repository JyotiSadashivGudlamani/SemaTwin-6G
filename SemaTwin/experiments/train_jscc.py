import os
import sys
import random
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob

from models.deep_jscc import DeepJSCCModel
from data.dataset import CIFAR10_CLASSES


class FastLocalCIFAR10Dataset(Dataset):
    """
    Loads images from local CIFAR-10 directory without downloading.
    """
    def __init__(self, data_dir, split='train'):
        self.samples = []
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            split_dir = data_dir

        for cls_name in CIFAR10_CLASSES:
            cls_dir = os.path.join(split_dir, cls_name)
            if os.path.isdir(cls_dir):
                imgs = glob.glob(os.path.join(cls_dir, "*.png")) + glob.glob(os.path.join(cls_dir, "*.jpg"))
                for img_p in imgs:
                    self.samples.append(img_p)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p = self.samples[idx]
        pil_img = Image.open(img_p).convert("RGB")
        img_np = np.array(pil_img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.tensor(img_np, dtype=torch.float32)


def train_deep_jscc(epochs=3, batch_size=64, max_batches_per_epoch=200, lr=1e-3):
    print("============================================================", flush=True)
    print(" PHASE 8A: REAL PYTORCH DEEP JSCC TRAINING", flush=True)
    print("============================================================", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Compute Device: {device}", flush=True)

    # Dataset path
    data_dir = os.path.join(PROJECT_ROOT, "data", "cifar10")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(PROJECT_ROOT, "data")

    print(f" Loading Local CIFAR-10 Dataset from: {data_dir} (download=False)", flush=True)
    dataset = FastLocalCIFAR10Dataset(data_dir, split='train')
    
    if len(dataset) == 0:
        print("[WARNING] Local images empty, generating synthetic dataset for training", flush=True)
        syn_x = np.random.uniform(0.0, 1.0, (1000, 3, 32, 32)).astype(np.float32)
        dataset = torch.tensor(syn_x, dtype=torch.float32)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model & Optimizer
    model = DeepJSCCModel(latent_channels=16).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    print(f" Starting PyTorch Model Training ({epochs} Epochs)...", flush=True)

    final_loss = 0.0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_batches = 0

        for batch_idx, batch_x in enumerate(dataloader):
            if max_batches_per_epoch and batch_idx >= max_batches_per_epoch:
                break

            if isinstance(batch_x, list) or isinstance(batch_x, tuple):
                batch_x = batch_x[0]

            batch_x = batch_x.to(device)

            # Random SNR between 5 dB and 25 dB for robust training
            snr_db = random.uniform(5.0, 25.0)

            optimizer.zero_grad()
            reconstructed, _, _ = model(batch_x, snr_db=snr_db)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        final_loss = avg_loss
        print(f" Epoch [{epoch}/{epochs}] - Loss (MSE): {avg_loss:.6f}", flush=True)

    # Save Checkpoint
    output_dir = os.path.join(PROJECT_ROOT, "results", "models")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "deep_jscc_cifar10.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n [SUCCESS] Trained PyTorch Model Checkpoint Saved To: {save_path}", flush=True)

    return {
        "epochs": epochs,
        "final_mse_loss": final_loss,
        "checkpoint_path": save_path,
        "device": str(device)
    }


if __name__ == "__main__":
    train_deep_jscc(epochs=3, batch_size=64, max_batches_per_epoch=200, lr=1e-3)
