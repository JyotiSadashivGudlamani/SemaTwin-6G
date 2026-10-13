import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from digital_twin.twin import DigitalTwinEngine
from communication.transmission import SemanticTransmissionPipeline


def test_phase3():
    dt_engine = DigitalTwinEngine()
    pipeline = SemanticTransmissionPipeline()

    # 1. Transmit packets to N3 and update N3 existing telemetry
    pipeline.transmit_packets(destination_node_id="N3", num_packets=100, dt_engine=dt_engine)

    # 2. Read N3 telemetry again from existing Digital Twin state
    n3_telemetry = dt_engine.state.node_telemetry.get("N3", {})

    packets_sent = n3_telemetry.get("packets_sent", 0)
    packets_received = n3_telemetry.get("packets_received", 0)
    packets_lost = n3_telemetry.get("packets_lost", 0)
    packet_loss_pct = n3_telemetry.get("packet_loss_pct", 0.0)

    # 3. Verify telemetry updated
    is_updated = (packets_sent == 100 and packets_received + packets_lost == 100)

    print("========================================")
    print("PHASE 3 NODE TELEMETRY TEST")
    print("========================================")
    print(f"Node: {n3_telemetry.get('node_id', 'N3')}")
    print(f"Packets Sent: {packets_sent}")
    print(f"Packets Received: {packets_received}")
    print(f"Packets Lost: {packets_lost}")
    print(f"Packet Loss: {packet_loss_pct:.2f}%")
    print(f"Telemetry Updated: {'YES' if is_updated else 'NO'}")
    print("========================================")


if __name__ == "__main__":
    test_phase3()
