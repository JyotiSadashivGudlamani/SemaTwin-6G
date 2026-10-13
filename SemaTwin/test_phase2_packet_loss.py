import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from communication.transmission import SemanticTransmissionPipeline


def test_phase2():
    pipeline = SemanticTransmissionPipeline()
    result = pipeline.transmit_packets(destination_node_id="N3", num_packets=100)

    print("========================================")
    print("PHASE 2 PACKET LOSS TEST")
    print("========================================")
    print(f"Source Server: {result['source_server']}")
    print(f"Destination Node: {result['destination_node']}")
    print(f"Packets Sent: {result['packets_sent']}")
    print(f"Packets Received: {result['packets_received']}")
    print(f"Packets Lost: {result['packets_lost']}")
    print(f"Packet Loss: {result['packet_loss_pct']:.2f}%")
    print("========================================")


if __name__ == "__main__":
    test_phase2()
