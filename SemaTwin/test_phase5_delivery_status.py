import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from digital_twin.twin import DigitalTwinEngine
from communication.transmission import SemanticTransmissionPipeline


def test_phase5():
    dt_engine = DigitalTwinEngine()
    pipeline = SemanticTransmissionPipeline()

    # Transmit packets to N3 and update N3 existing state in Digital Twin
    result = pipeline.transmit_packets(destination_node_id="N3", num_packets=100, dt_engine=dt_engine)

    # Read updated delivery_status from existing Digital Twin telemetry state
    n3_state = dt_engine.state.node_telemetry.get("N3", {})
    delivery_status = n3_state.get("delivery_status", result.get("delivery_status", "UNKNOWN"))

    print("========================================")
    print("PHASE 5 DELIVERY STATUS TEST")
    print("========================================")
    print(f"Node: {result['destination_node']}")
    print(f"Packets Sent: {result['packets_sent']}")
    print(f"Packets Received: {result['packets_received']}")
    print(f"Packets Lost: {result['packets_lost']}")
    print(f"Packet Loss: {result['packet_loss_pct']:.2f}%")
    print(f"Delivery Status: {delivery_status}")
    print("========================================")


if __name__ == "__main__":
    test_phase5()
