import sys
import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from digital_twin.twin import DigitalTwinEngine
from communication.transmission import SemanticTransmissionPipeline
import numpy as np

dt_engine = DigitalTwinEngine()
pipeline = SemanticTransmissionPipeline()

# Random dummy CIFAR-10 image [32, 32, 3]
dummy_img = np.random.uniform(0, 1, (32, 32, 3)).astype(np.float32)

nodes = ["N1", "N2", "N3"]
results = []

for n in nodes:
    telemetry = dt_engine.state.node_telemetry[n]
    res = pipeline.transmit_real_jscc(
        pil_or_np_img=dummy_img,
        destination_node_id=n,
        node_telemetry=telemetry,
        dt_engine=dt_engine
    )
    results.append(res)

rows = []
for r in results:
    nid = r["destination_node"]
    tel = dt_engine.state.node_telemetry[nid]
    rows.append({
        "Node ID": nid,
        "Server": r["source_server"],
        "Packets Sent": r["packets_sent"],
        "Packets Received": r["packets_received"],
        "Packets Lost": r["packets_lost"],
        "Packet Loss %": f"{r['packet_loss_pct']:.2f}%",
        "SNR": f"{r['snr_db']:.2f} dB",
        "Latency": f"{tel['latency_ms']:.2f} ms",
        "Bandwidth": f"{tel['bandwidth_mbps']:.2f} Mbps",
        "Health/Condition": tel["health"]
    })

df = pd.DataFrame(rows)
print("=== MULTI-NODE TRANSMISSION TEST RESULT ===")
print(df.to_string(index=False))
