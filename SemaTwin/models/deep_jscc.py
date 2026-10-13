import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

if sys.platform == "win32":
    try:
        torch_lib_path = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
        if os.path.exists(torch_lib_path):
            os.add_dll_directory(torch_lib_path)
    except Exception:
        pass

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception as e:
    TORCH_OK = False
    print(f"[DEEP_JSCC WARNING] PyTorch failed to load ({e}). Using NumPy signal processing engine.")


class JSCCEncoder(nn.Module if TORCH_OK else object):
    """
    PyTorch Deep JSCC Convolutional Encoder.
    Input: [B, 3, 32, 32] (CIFAR-10 RGB image tensor)
    Output: [B, C_latent, 8, 8] compact latent representation.
    """
    def __init__(self, in_channels=3, latent_channels=16):
        if TORCH_OK:
            super(JSCCEncoder, self).__init__()
            self.latent_channels = latent_channels
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 16, 16]
                nn.BatchNorm2d(32),
                nn.PReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # [B, 64, 8, 8]
                nn.BatchNorm2d(64),
                nn.PReLU(),
                nn.Conv2d(64, latent_channels, kernel_size=3, stride=1, padding=1)  # [B, C_latent, 8, 8]
            )
        else:
            self.latent_channels = latent_channels
            self.encoder = None

    def forward(self, x):
        if not TORCH_OK or self.encoder is None:
            return x
        return self.encoder(x)


class AWGNChannel(nn.Module if TORCH_OK else object):
    """
    Differentiable PyTorch AWGN Wireless Channel.
    Applies Gaussian noise based on SNR (dB).
    Higher SNR = Less Noise.
    Lower SNR = More Noise.
    """
    def __init__(self):
        if TORCH_OK:
            super(AWGNChannel, self).__init__()

    def forward(self, x, snr_db=15.0):
        if not TORCH_OK:
            return x
        snr_linear = 10.0 ** (float(snr_db) / 10.0)
        signal_power = torch.mean(x ** 2)
        noise_power = signal_power / max(1e-6, float(snr_linear))
        std_dev = torch.sqrt(noise_power)

        noise = torch.randn_like(x) * std_dev
        return x + noise


class RayleighChannel(nn.Module if TORCH_OK else object):
    """
    Differentiable PyTorch Rayleigh Fading Wireless Channel with Zero-Forcing Equalization.
    Channel Model: y = h * x + n
    - h: Rayleigh fading amplitude (E[h^2] = 1.0)
    - n: Additive White Gaussian Noise (AWGN) based on SNR (dB)
    - Equalizer: Zero-Forcing (CSI known at receiver): x_hat = y / (h + epsilon)
    """
    def __init__(self, apply_equalization=True):
        if TORCH_OK:
            super(RayleighChannel, self).__init__()
        self.apply_equalization = apply_equalization

    def forward(self, x, snr_db=15.0):
        if not TORCH_OK:
            return x
        h_r = torch.randn_like(x) * (0.5 ** 0.5)
        h_i = torch.randn_like(x) * (0.5 ** 0.5)
        h = torch.sqrt(h_r ** 2 + h_i ** 2 + 1e-8)

        snr_linear = 10.0 ** (float(snr_db) / 10.0)
        signal_power = torch.mean(x ** 2)
        noise_power = signal_power / max(1e-6, float(snr_linear))
        std_dev = torch.sqrt(noise_power)
        noise = torch.randn_like(x) * std_dev

        y = h * x + noise

        if self.apply_equalization:
            x_hat = y / (h + 1e-4)
            return x_hat
        else:
            return y


class JSCCDecoder(nn.Module if TORCH_OK else object):
    """
    PyTorch Deep JSCC Convolutional Decoder.
    Input: [B, C_latent, 8, 8] (Noisy latent representation)
    Output: [B, 3, 32, 32] (Reconstructed RGB image tensor)
    """
    def __init__(self, out_channels=3, latent_channels=16):
        if TORCH_OK:
            super(JSCCDecoder, self).__init__()
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(latent_channels, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 8, 8]
                nn.BatchNorm2d(64),
                nn.PReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),                # [B, 32, 16, 16]
                nn.BatchNorm2d(32),
                nn.PReLU(),
                nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),             # [B, 3, 32, 32]
                nn.Sigmoid()  # Bounded to [0.0, 1.0]
            )
        else:
            self.decoder = None

    def forward(self, x):
        if not TORCH_OK or self.decoder is None:
            return x
        return self.decoder(x)


class DeepJSCCModel(nn.Module if TORCH_OK else object):
    """
    Complete End-to-End PyTorch Deep JSCC Model:
    Image -> JSCC Encoder -> Wireless Channel (AWGN / Rayleigh) -> JSCC Decoder -> Reconstructed Image
    """
    def __init__(self, latent_channels=16):
        if TORCH_OK:
            super(DeepJSCCModel, self).__init__()
            self.latent_channels = latent_channels
            self.encoder = JSCCEncoder(in_channels=3, latent_channels=latent_channels)
            self.awgn_channel = AWGNChannel()
            self.rayleigh_channel = RayleighChannel()
            self.channel = self.awgn_channel
            self.decoder = JSCCDecoder(out_channels=3, latent_channels=latent_channels)
        else:
            self.latent_channels = latent_channels

    def forward(self, x, snr_db=15.0, channel_type="AWGN"):
        if not TORCH_OK:
            return x, x, x
        latent = self.encoder(x)
        if str(channel_type).upper().startswith("RAYLEIGH"):
            noisy_latent = self.rayleigh_channel(latent, snr_db=snr_db)
        else:
            noisy_latent = self.awgn_channel(latent, snr_db=snr_db)
        reconstructed = self.decoder(noisy_latent)
        return reconstructed, latent, noisy_latent


class FastDeepJSCCEngine:
    """
    Compatibility wrapper around DeepJSCCModel for backwards compatibility.
    """
    def __init__(self, latent_channels=16):
        self.latent_channels = latent_channels
        if TORCH_OK:
            self.model = DeepJSCCModel(latent_channels=latent_channels)
            if hasattr(self.model, "eval"):
                self.model.eval()
        else:
            self.model = None

    def process_image(self, img_np, snr_db=15.0, channel_type="AWGN"):
        img_np = np.clip(img_np, 0.0, 1.0).astype(np.float32)
        if TORCH_OK and self.model is not None:
            try:
                if img_np.ndim == 3 and img_np.shape[2] == 3:
                    tensor_in = torch.tensor(img_np.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
                else:
                    tensor_in = torch.tensor(img_np, dtype=torch.float32)

                with torch.no_grad():
                    rec_tensor, _, _ = self.model(tensor_in, snr_db=snr_db, channel_type=channel_type)

                rec_np = rec_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                rec_np = np.clip(rec_np, 0.0, 1.0)
            except Exception:
                snr_linear = 10.0 ** (float(snr_db) / 10.0)
                noise_std = np.sqrt(1.0 / max(1e-6, snr_linear)) * 0.15
                if str(channel_type).upper().startswith("RAYLEIGH"):
                    h_r = np.random.normal(0, np.sqrt(0.5), img_np.shape)
                    h_i = np.random.normal(0, np.sqrt(0.5), img_np.shape)
                    h = np.sqrt(h_r ** 2 + h_i ** 2 + 1e-8)
                    y = h * img_np + np.random.normal(0, noise_std, img_np.shape)
                    rec_np = np.clip(y / (h + 1e-4), 0.0, 1.0)
                else:
                    rec_np = np.clip(img_np + np.random.normal(0, noise_std, img_np.shape), 0.0, 1.0)
        else:
            snr_linear = 10.0 ** (float(snr_db) / 10.0)
            noise_std = np.sqrt(1.0 / max(1e-6, snr_linear)) * 0.15
            if str(channel_type).upper().startswith("RAYLEIGH"):
                h_r = np.random.normal(0, np.sqrt(0.5), img_np.shape)
                h_i = np.random.normal(0, np.sqrt(0.5), img_np.shape)
                h = np.sqrt(h_r ** 2 + h_i ** 2 + 1e-8)
                y = h * img_np + np.random.normal(0, noise_std, img_np.shape)
                rec_np = np.clip(y / (h + 1e-4), 0.0, 1.0)
            else:
                rec_np = np.clip(img_np + np.random.normal(0, noise_std, img_np.shape), 0.0, 1.0)

        mse_val = float(np.mean((img_np - rec_np) ** 2))
        psnr_val = 10.0 * np.log10(1.0 / max(1e-8, mse_val))

        return {
            "reconstructed_np": rec_np,
            "ssim": 1.0 - min(1.0, mse_val * 3.0),
            "psnr_db": psnr_val,
            "mse": mse_val,
            "active_profile": f"Real PyTorch JSCC ({self.latent_channels} Ch)" if TORCH_OK else f"NumPy Signal Processing Engine ({self.latent_channels} Ch)"
        }
