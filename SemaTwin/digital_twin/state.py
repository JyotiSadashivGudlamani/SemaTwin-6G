import time
import random


class DigitalTwinState:
    """
    Phase 7 & 9: Digital Twin State Mirror, Telemetry Stream & Time-Series SNR History.
    Maintains virtual state, real-time telemetry, and historical SNR series for:
    - Virtual Server 1 -> N1, N2, N3, N4, N5
    - Virtual Server 2 -> N6, N7, N8, N9, N10
    """
    def __init__(self):
        self.node_telemetry = {}
        self.snr_history = {f"N{i}": [] for i in range(1, 11)}
        self.last_update = time.strftime("%H:%M:%S", time.localtime())
        self._initialize_telemetry()

    def _initialize_telemetry(self):
        for i in range(1, 11):
            n_id = f"N{i}"
            server_id = "Server 1" if i <= 5 else "Server 2"
            initial_snr = round(random.uniform(15.0, 20.0), 2)
            self.node_telemetry[n_id] = {
                "node_id": n_id,
                "server_id": server_id,
                "status": "ACTIVE",
                "snr_db": initial_snr,
                "latency_ms": round(random.uniform(8.0, 15.0), 2),
                "bandwidth_mbps": round(random.uniform(80.0, 100.0), 2),
                "packet_loss_pct": round(random.uniform(0.1, 0.5), 2),
                "packets_sent": 100,
                "packets_received": 100,
                "packets_lost": 0,
                "delivery_status": "DELIVERED",
                "semantic_meaning": "DOG",
                "health": "EXCELLENT"
            }
            # Populate initial history seed
            if not self.snr_history[n_id]:
                base_snr = initial_snr
                self.snr_history[n_id] = [
                    round(max(5.0, base_snr + random.uniform(-1.5, 1.5)), 2) for _ in range(8)
                ] + [initial_snr]
        self.last_update = time.strftime("%H:%M:%S", time.localtime())

    def update_node_packet_loss(self, node_id, packets_sent, packets_received, packet_loss_pct, semantic_meaning=None):
        """Updates packet loss statistics, delivery status, and semantic content for a specific node in telemetry."""
        if node_id in self.node_telemetry:
            data = self.node_telemetry[node_id]
            data["packets_sent"] = int(packets_sent)
            data["packets_received"] = int(packets_received)
            data["packets_lost"] = int(packets_sent - packets_received)
            data["packet_loss_pct"] = round(float(packet_loss_pct), 2)
            if semantic_meaning:
                data["semantic_meaning"] = str(semantic_meaning).upper()

            # Derive Node Delivery Status based on actual packet delivery
            if data["status"] == "INACTIVE" or data["packets_received"] == 0:
                data["delivery_status"] = "FAILED"
            elif data["packets_lost"] == 0:
                data["delivery_status"] = "DELIVERED"
            else:
                data["delivery_status"] = "PARTIAL"

            val_snr = data["snr_db"]
            loss = data["packet_loss_pct"]

            if data["status"] == "INACTIVE":
                data["health"] = "INACTIVE"
            elif val_snr >= 18.0 and loss < 1.0:
                data["health"] = "EXCELLENT"
            elif val_snr >= 12.0 and loss < 3.0:
                data["health"] = "GOOD"
            elif val_snr >= 7.0:
                data["health"] = "DEGRADED"
            else:
                data["health"] = "POOR"

            self.last_update = time.strftime("%H:%M:%S", time.localtime())

    def update_telemetry(self, scenario="normal"):
        """
        Updates network telemetry values across all nodes based on selected condition scenario.
        Appends new SNR values to time-series history for LSTM forecasting.
        """
        for n_id, data in self.node_telemetry.items():
            if data["status"] == "INACTIVE":
                data["snr_db"] = 0.0
                data["latency_ms"] = 999.0
                data["bandwidth_mbps"] = 0.0
                data["packet_loss_pct"] = 100.0
                data["health"] = "INACTIVE"
                self.snr_history[n_id].append(0.0)
                continue

            if scenario == "normal":
                snr = random.uniform(15.0, 20.0)
                lat = random.uniform(8.0, 15.0)
                bw = random.uniform(80.0, 100.0)
                loss = random.uniform(0.1, 0.5)
            elif scenario == "degrading":
                snr = random.uniform(8.0, 14.0)
                lat = random.uniform(20.0, 40.0)
                bw = random.uniform(30.0, 60.0)
                loss = random.uniform(1.0, 3.0)
            elif scenario == "high_interference":
                snr = random.uniform(2.0, 7.0)
                lat = random.uniform(50.0, 100.0)
                bw = random.uniform(5.0, 20.0)
                loss = random.uniform(5.0, 12.0)
            else:  # random_fluctuation
                snr = max(1.0, min(20.0, data["snr_db"] + random.uniform(-2.5, 2.5)))
                lat = max(2.0, min(200.0, data["latency_ms"] + random.uniform(-3.0, 3.0)))
                bw = max(1.0, min(120.0, data["bandwidth_mbps"] + random.uniform(-5.0, 5.0)))
                loss = max(0.0, min(20.0, data["packet_loss_pct"] + random.uniform(-0.2, 0.2)))

            val_snr = round(snr, 2)
            data["snr_db"] = val_snr
            data["latency_ms"] = round(lat, 2)
            data["bandwidth_mbps"] = round(bw, 2)
            data["packet_loss_pct"] = round(loss, 2)

            self.snr_history[n_id].append(val_snr)
            if len(self.snr_history[n_id]) > 50:
                self.snr_history[n_id] = self.snr_history[n_id][-50:]

            # Assign Health status label
            if val_snr >= 18.0 and loss < 1.0:
                data["health"] = "EXCELLENT"
            elif val_snr >= 12.0 and loss < 3.0:
                data["health"] = "GOOD"
            elif val_snr >= 7.0:
                data["health"] = "DEGRADED"
            else:
                data["health"] = "POOR"

        self.last_update = time.strftime("%H:%M:%S", time.localtime())

    def get_node_snr_series(self, node_id, window=10):
        """Returns the recent SNR time-series list for a given node."""
        clean_id = node_id.replace("Node ", "N").replace("Node", "N").strip() if isinstance(node_id, str) else node_id
        if clean_id in self.snr_history and self.snr_history[clean_id]:
            return self.snr_history[clean_id][-window:]
        if node_id in self.snr_history and self.snr_history[node_id]:
            return self.snr_history[node_id][-window:]
        return [18.0] * window

    def toggle_node_state(self, node_id):
        if node_id in self.node_telemetry:
            current = self.node_telemetry[node_id]["status"]
            new_st = "INACTIVE" if current == "ACTIVE" else "ACTIVE"
            self.node_telemetry[node_id]["status"] = new_st
            
            if new_st == "INACTIVE":
                self.node_telemetry[node_id]["snr_db"] = 0.0
                self.node_telemetry[node_id]["latency_ms"] = 999.0
                self.node_telemetry[node_id]["bandwidth_mbps"] = 0.0
                self.node_telemetry[node_id]["packet_loss_pct"] = 100.0
                self.node_telemetry[node_id]["packets_sent"] = 100
                self.node_telemetry[node_id]["packets_received"] = 0
                self.node_telemetry[node_id]["packets_lost"] = 100
                self.node_telemetry[node_id]["health"] = "INACTIVE"
                self.snr_history[node_id].append(0.0)
            else:
                val_snr = round(random.uniform(15.0, 22.0), 2)
                self.node_telemetry[node_id]["snr_db"] = val_snr
                self.node_telemetry[node_id]["latency_ms"] = round(random.uniform(10.0, 20.0), 2)
                self.node_telemetry[node_id]["bandwidth_mbps"] = round(random.uniform(70.0, 95.0), 2)
                self.node_telemetry[node_id]["packet_loss_pct"] = round(random.uniform(0.1, 0.8), 2)
                self.node_telemetry[node_id]["packets_sent"] = 100
                self.node_telemetry[node_id]["packets_received"] = 99
                self.node_telemetry[node_id]["packets_lost"] = 1
                self.node_telemetry[node_id]["health"] = "EXCELLENT"
                self.snr_history[node_id].append(val_snr)

            self.last_update = time.strftime("%H:%M:%S", time.localtime())
            return new_st
        return "UNKNOWN"

    def set_node_state(self, node_id, state):
        if node_id in self.node_telemetry:
            target_st = "ACTIVE" if str(state).upper() == "ACTIVE" else "INACTIVE"
            if self.node_telemetry[node_id]["status"] != target_st:
                self.toggle_node_state(node_id)

    @property
    def node_states(self):
        return {nid: data["status"] for nid, data in self.node_telemetry.items()}

    def get_summary(self):
        telemetry_list = list(self.node_telemetry.values())
        active_nodes = [t for t in telemetry_list if t["status"] == "ACTIVE"]
        inactive_nodes = [t for t in telemetry_list if t["status"] == "INACTIVE"]

        avg_snr = round(sum(t["snr_db"] for t in active_nodes) / len(active_nodes), 2) if active_nodes else 0.0
        avg_lat = round(sum(t["latency_ms"] for t in active_nodes) / len(active_nodes), 2) if active_nodes else 0.0
        avg_bw = round(sum(t["bandwidth_mbps"] for t in active_nodes) / len(active_nodes), 2) if active_nodes else 0.0
        avg_loss = round(sum(t["packet_loss_pct"] for t in active_nodes) / len(active_nodes), 2) if active_nodes else 100.0

        return {
            "total_virtual_nodes": len(self.node_telemetry),
            "active_virtual_nodes": len(active_nodes),
            "inactive_virtual_nodes": len(inactive_nodes),
            "avg_snr_db": avg_snr,
            "avg_latency_ms": avg_lat,
            "avg_bandwidth_mbps": avg_bw,
            "avg_packet_loss_pct": avg_loss,
            "last_update": self.last_update,
            "node_states": self.node_states,
            "telemetry_list": telemetry_list
        }
