class ProactiveAdaptiveController:
    """
    Proactive Adaptive Communication Controller.
    Selects communication profile (JSCC-A High Quality vs JSCC-B Robust) based on Digital Twin LSTM forecasts.
    """
    def __init__(self, snr_threshold_db=10.0):
        self.snr_threshold_db = snr_threshold_db

    def select_profile(self, predicted_snr_db):
        if predicted_snr_db >= self.snr_threshold_db:
            return {
                "profile": "JSCC-A (High Quality)",
                "channels": 16,
                "mode": "HIGH_FIDELITY",
                "description": "High predicted SNR (>= 10 dB). Allocating 16 channels for maximum visual detail."
            }
        else:
            return {
                "profile": "JSCC-B (Robust)",
                "channels": 4,
                "mode": "HIGH_COMPRESSION",
                "description": "Low predicted SNR (< 10 dB). Swapping to 4-channel robust mode to prevent transmission collapse."
            }
