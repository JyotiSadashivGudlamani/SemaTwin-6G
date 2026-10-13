from models.dt_lstm import DigitalTwinLSTMPredictor

class DigitalTwinPredictor:
    """
    Digital Twin AI Predictor Integration.
    Consumes time-series telemetry from Digital Twin State and invokes LSTM model to forecast SNR.
    """
    def __init__(self):
        self.lstm_model = DigitalTwinLSTMPredictor()

    def predict_node_future_snr(self, dt_state, node_id, window=10):
        snr_history = dt_state.get_node_snr_series(node_id, window=window)
        current_snr = snr_history[-1] if len(snr_history) > 0 else 15.0
        predicted_snr = self.lstm_model.predict_future_snr(snr_history)

        condition = "GOOD" if predicted_snr >= 10.0 else ("MODERATE" if predicted_snr >= 3.0 else "POOR")
        
        return {
            "node_id": node_id,
            "historical_snr": snr_history,
            "current_snr": current_snr,
            "predicted_future_snr": predicted_snr,
            "predicted_condition": condition,
            "warning": predicted_snr < 10.0
        }
