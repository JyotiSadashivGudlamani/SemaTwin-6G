import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from digital_twin.twin import DigitalTwinEngine
from communication.transmission import SemanticTransmissionPipeline


def test_phase4():
    # 1. Initialize existing Digital Twin Engine and Communication Pipeline
    dt_engine = DigitalTwinEngine()
    pipeline = SemanticTransmissionPipeline()

    # 2. Read Digital Twin state BEFORE transmission
    before_telemetry = dict(dt_engine.state.node_telemetry.get("N3", {}))
    before_loss = before_telemetry.get("packet_loss_pct", 0.0)

    # 3. Transmit packets to N3 (Communication -> Packet result -> Telemetry update)
    res = pipeline.transmit_packets(destination_node_id="N3", num_packets=100, dt_engine=dt_engine)

    # 4. Read Digital Twin state AFTER transmission
    after_telemetry = dt_engine.state.node_telemetry.get("N3", {})
    dt_packet_loss = after_telemetry.get("packet_loss_pct", 0.0)

    # 5. Confirm that Digital Twin state reflects exact transmission packet loss result
    feedback_success = (abs(dt_packet_loss - res["packet_loss_pct"]) < 1e-5)

    print("========================================")
    print("PHASE 4 DIGITAL TWIN FEEDBACK TEST")
    print("========================================")
    print(f"Destination: {res['destination_node']}")
    print(f"Packets Sent: {res['packets_sent']}")
    print(f"Packets Received: {res['packets_received']}")
    print(f"Packets Lost: {res['packets_lost']}")
    print(f"Packet Loss: {res['packet_loss_pct']:.2f}%")
    print("")
    print(f"Digital Twin N3 Packet Loss: {dt_packet_loss:.2f}%")
    print(f"Digital Twin Feedback: {'SUCCESS' if feedback_success else 'FAILED'}")
    print("========================================")

    return before_loss, dt_packet_loss, res


if __name__ == "__main__":
    test_phase4()
