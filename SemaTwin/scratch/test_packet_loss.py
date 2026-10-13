import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from digital_twin.twin import DigitalTwinEngine
from communication.transmission import SemanticTransmissionPipeline

dt_engine = DigitalTwinEngine()
pipeline = SemanticTransmissionPipeline()

dest_node = "N3"
dt_engine.update_telemetry(scenario="high_interference")
target_telemetry = dt_engine.state.node_telemetry[dest_node]
dummy_img = np.random.uniform(0, 1, (32, 32, 3)).astype(np.float32)

print("BEFORE TRANSMISSION (HIGH INTERFERENCE):")
print(f"SNR: {target_telemetry['snr_db']} dB, Initial Loss %: {target_telemetry['packet_loss_pct']}%")

res = pipeline.transmit_real_jscc(dummy_img, dest_node, target_telemetry, dt_engine=dt_engine)

print("\nTRANSMISSION PACKET CALCULATION:")
print(f"Packets Sent:     {res['packets_sent']}")
print(f"Packets Received: {res['packets_received']}")
print(f"Packets Lost:     {res['packets_lost']}")
print(f"Packet Loss %:    {res['packet_loss_pct']:.2f}%")

print("\nAFTER TRANSMISSION DIGITAL TWIN TELEMETRY:")
updated_telemetry = dt_engine.state.node_telemetry[dest_node]
print(f"Node ID:          {updated_telemetry['node_id']}")
print(f"SNR (dB):         {updated_telemetry['snr_db']} dB")
print(f"Packet Loss (%):  {updated_telemetry['packet_loss_pct']}%")
print(f"Node Health:      {updated_telemetry['health']}")
