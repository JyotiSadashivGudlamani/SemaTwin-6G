import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.dt_lstm import DigitalTwinLSTMPredictor

def train_or_evaluate_lstm():
    print("============================================================")
    print(" EXPERIMENT: DIGITAL TWIN LSTM PREDICTOR EVALUATION")
    print("============================================================")
    predictor = DigitalTwinLSTMPredictor()
    sample_snr = [14.0, 13.5, 12.0, 10.5, 9.0]
    pred = predictor.predict_future_snr(sample_snr)
    print(f"[OK] Historical SNR: {sample_snr} -> Predicted Future SNR: {pred} dB")
    print("[SUCCESS] LSTM Predictor Experiment Script Ready.")

if __name__ == "__main__":
    train_or_evaluate_lstm()
