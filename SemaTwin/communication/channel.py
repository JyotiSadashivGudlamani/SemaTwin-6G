import random
import math
import hashlib


class SimulatedChannel:
    """
    Phase 8: Simulated 6G Wireless Channel.
    Simulates semantic packet propagation over wireless channel using live node telemetry:
    - SNR (dB)
    - Latency (ms)
    - Bandwidth (Mbps)
    - Packet Loss (%)
    - Node Status (ACTIVE / INACTIVE)
    """
    def __init__(self):
        pass

    def propagate(self, semantic_meaning, telemetry):
        """
        Simulates channel propagation for semantic meaning.
        
        Returns:
            dict containing:
            - transmission_status: "SUCCESSFUL", "DEGRADED", or "FAILED"
            - received_semantic_meaning: Received class name or failure state
            - effective_latency_ms: Calculated total propagation latency
            - packet_loss_occurred: bool
        """
        status = telemetry.get("status", "ACTIVE")
        snr_db = float(telemetry.get("snr_db", 20.0))
        latency_ms = float(telemetry.get("latency_ms", 10.0))
        bandwidth_mbps = float(telemetry.get("bandwidth_mbps", 100.0))
        packet_loss_pct = float(telemetry.get("packet_loss_pct", 0.0))

        # Check 1: Inactive / Offline Node
        if status == "INACTIVE":
            return {
                "transmission_status": "FAILED (NODE OFFLINE)",
                "received_semantic_meaning": "NO SIGNAL (OFFLINE)",
                "effective_latency_ms": 999.0,
                "packet_loss_occurred": True,
                "detail": "Target destination node is INACTIVE/OFFLINE."
            }

        # Check 2: High Packet Loss or Extremely Low SNR
        if packet_loss_pct >= 50.0 or snr_db < 3.0:
            return {
                "transmission_status": "FAILED (PACKET DROPPED)",
                "received_semantic_meaning": "CORRUPTED / DROPPED",
                "effective_latency_ms": round(latency_ms * 2.5, 2),
                "packet_loss_occurred": True,
                "detail": f"Packet dropped due to channel loss ({packet_loss_pct}%) or severe noise (SNR {snr_db} dB)."
            }

        # Check 3: Degraded Channel (Moderate SNR or High Latency)
        if snr_db < 10.0 or packet_loss_pct > 3.0:
            return {
                "transmission_status": "SUCCESSFUL (DEGRADED LINK)",
                "received_semantic_meaning": f"{semantic_meaning} (NOISY)",
                "effective_latency_ms": round(latency_ms * 1.4, 2),
                "packet_loss_occurred": False,
                "detail": f"Semantic meaning received over degraded wireless channel (SNR {snr_db} dB)."
            }

        # Check 4: Excellent / Normal Channel
        return {
            "transmission_status": "SUCCESSFUL (EXCELLENT LINK)",
            "received_semantic_meaning": semantic_meaning,
            "effective_latency_ms": round(latency_ms, 2),
            "packet_loss_occurred": False,
            "detail": f"Semantic meaning transmitted cleanly over 6G channel (SNR {snr_db} dB, Latency {latency_ms} ms)."
        }

    def simulate_packet_transmission(self, num_packets=100, snr_db=20.0, baseline_loss_pct=0.0, status="ACTIVE", channel_type=None):
        """
        Simulates sending N packets over wireless channel.
        Calculates packet loss deterministically based on snr_db, channel_type, and node status.
        """
        if status == "INACTIVE":
            return {
                "packets_sent": num_packets,
                "packets_received": 0,
                "packets_lost": num_packets,
                "packet_loss_pct": 100.0
            }

        snr_val = float(snr_db)
        packets_sent = int(num_packets)

        if str(channel_type).upper().startswith("RAYLEIGH"):
            # Fully deterministic Rayleigh fading penalty derived directly from SNR (no random values)
            awgn_loss_pct = max(0.0, min(100.0, (20.0 - snr_val) * 4.25))
            fading_penalty = max(0.0, (20.0 - snr_val) * 1.75)
            loss_pct = max(0.0, min(100.0, awgn_loss_pct + fading_penalty))
        else:
            # Baseline AWGN Channel Model: Deterministic linear SNR loss curve
            loss_pct = max(0.0, min(100.0, (20.0 - snr_val) * 4.25))

        packets_lost = int(round((loss_pct / 100.0) * packets_sent))

        packets_lost = min(packets_sent, max(0, packets_lost))
        packets_received = packets_sent - packets_lost
        actual_loss_pct = round((packets_lost / packets_sent) * 100.0, 2)

        return {
            "packets_sent": packets_sent,
            "packets_received": packets_received,
            "packets_lost": packets_lost,
            "packet_loss_pct": actual_loss_pct
        }
