import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from digital_twin.twin import DigitalTwinEngine
from communication.transmission import SemanticTransmissionPipeline


def run_e2e_verification():
    # 1. Initialize existing Digital Twin Engine and Communication Pipeline
    dt_engine = DigitalTwinEngine()
    pipeline = SemanticTransmissionPipeline()

    # 2. Execute End-to-End Packet Transmission to N3
    result = pipeline.transmit_packets(destination_node_id="N3", num_packets=100, dt_engine=dt_engine)

    # 3. Read N3 telemetry from Digital Twin State post-transmission
    n3_telemetry = dt_engine.state.node_telemetry.get("N3", {})

    packets_sent = result["packets_sent"]
    packets_received = result["packets_received"]
    packets_lost = result["packets_lost"]
    packet_loss_pct = result["packet_loss_pct"]
    delivery_status = result["delivery_status"]

    # 4. Verify mathematical invariants
    assert packets_sent == packets_received + packets_lost, "packets_sent != packets_received + packets_lost"
    expected_loss_pct = round((packets_lost / packets_sent) * 100.0, 2)
    assert abs(packet_loss_pct - expected_loss_pct) < 1e-4, "packet_loss_pct calculation mismatch"

    # 5. Verify telemetry and Digital Twin state updates
    telemetry_updated = (
        n3_telemetry.get("packets_sent") == packets_sent and
        n3_telemetry.get("packets_received") == packets_received and
        n3_telemetry.get("packets_lost") == packets_lost and
        n3_telemetry.get("delivery_status") == delivery_status
    )
    
    dt_updated = (
        abs(n3_telemetry.get("packet_loss_pct", -1.0) - packet_loss_pct) < 1e-4
    )

    print("==================================================")
    print("END-TO-END COMMUNICATION -> DIGITAL TWIN TEST")
    print("==================================================")
    print(f"Source Server: {result['source_server']}")
    print(f"Destination Node: {result['destination_node']}")
    print("")
    print(f"Packets Sent: {packets_sent}")
    print(f"Packets Received: {packets_received}")
    print(f"Packets Lost: {packets_lost}")
    print(f"Packet Loss: {packet_loss_pct:.2f}%")
    print(f"Delivery Status: {delivery_status}")
    print("")
    print(f"N3 Telemetry Updated: {'YES' if telemetry_updated else 'NO'}")
    print(f"Digital Twin Updated: {'YES' if dt_updated else 'NO'}")
    print("==================================================")


if __name__ == "__main__":
    run_e2e_verification()
