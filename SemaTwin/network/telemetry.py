import numpy as np

class TelemetryGenerator:
    """
    Simulated 6G Telemetry Stream Generator for SNR, latency, bandwidth, and packet loss.
    Produces realistic physical noise values for edge nodes.
    """
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.time_step = 0

    def generate_node_telemetry(self, node_id, distance_m=100.0, scenario="normal"):
        self.time_step += 1
        
        # Distance-based path loss base SNR
        base_snr = max(2.0, 25.0 - (distance_m / 20.0))
        noise = np.random.normal(0.0, 1.5)
        
        if scenario == "good":
            snr = min(30.0, base_snr + 8.0 + noise)
            latency = max(2.0, 5.0 + np.random.normal(0, 0.5))
            bandwidth = 50.0
            packet_loss = max(0.0, 0.01 + np.random.normal(0, 0.005))
        elif scenario == "degrading":
            # Degrading trend over time
            snr = max(-5.0, base_snr - (self.time_step * 1.2) + noise)
            latency = min(120.0, 20.0 + (self.time_step * 3.0))
            bandwidth = max(5.0, 30.0 - (self.time_step * 2.0))
            packet_loss = min(25.0, 0.5 + (self.time_step * 1.5))
        else:
            snr = max(1.0, base_snr + noise)
            latency = max(4.0, 12.0 + np.random.normal(0, 2.0))
            bandwidth = 20.0
            packet_loss = max(0.0, 0.1 + np.random.normal(0, 0.05))

        return {
            "node_id": node_id,
            "SNR_dB": round(float(snr), 2),
            "latency_ms": round(float(latency), 2),
            "bandwidth_MHz": round(float(bandwidth), 2),
            "packet_loss_pct": round(float(packet_loss), 2),
            "scenario": scenario,
            "time_step": self.time_step
        }

if __name__ == "__main__":
    gen = TelemetryGenerator()
    sample_t = gen.generate_node_telemetry("Node 1", distance_m=120.0)
    print(f"[OK] Telemetry Generator Test Passed: {sample_t}")
