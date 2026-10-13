import numpy as np

def compute_psnr(img1, img2):
    """Computes exact PSNR in dB between two normalized [0..1] numpy images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return round(float(10.0 * np.log10(1.0 / mse)), 2)

def compute_ssim(img1, img2):
    """Computes exact structural similarity index (SSIM) between two normalized images."""
    img1 = np.clip(img1, 0.0, 1.0)
    img2 = np.clip(img2, 0.0, 1.0)
    
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    var1 = np.var(img1)
    var2 = np.var(img2)
    cov = np.mean((img1 - mu1) * (img2 - mu2))

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    ssim_val = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
    return round(float(ssim_val), 3)

def compute_prediction_error(y_true, y_pred):
    """Computes MAE and RMSE for SNR predictions."""
    mae = float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))
    rmse = float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3)}
