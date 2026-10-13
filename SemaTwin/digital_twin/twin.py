import os
import sys
import plotly.graph_objects as go
import plotly.express as px

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib
import network.topology
importlib.reload(network.topology)
from network.topology import NetworkTopology
from digital_twin.state import DigitalTwinState


class DigitalTwinEngine:
    """
    Phase 7: Digital Twin Engine with Live Network Telemetry Stream.
    Maintains virtual mirror of 6G topology and tracks live telemetry:
    - SNR (dB)
    - Latency (ms)
    - Bandwidth (Mbps)
    - Packet Loss (%)
    - Node Status (ACTIVE / INACTIVE)
    """
    def __init__(self, topology=None):
        self.topology = topology or NetworkTopology()
        self.state = DigitalTwinState()

    def sync(self):
        """Synchronizes virtual twin state with underlying physical topology."""
        if hasattr(self.topology, "get_topology_summary"):
            summary = self.topology.get_topology_summary()
            for n in summary.get("server_1_nodes", []) + summary.get("server_2_nodes", []):
                n_id = n.get("node_id")
                if n_id and n_id in self.state.node_telemetry:
                    self.state.node_telemetry[n_id]["snr_db"] = n.get("snr_db", self.state.node_telemetry[n_id]["snr_db"])
                    self.state.node_telemetry[n_id]["status"] = n.get("status", self.state.node_telemetry[n_id]["status"])
        self.state.update_telemetry()
        return True

    def update_telemetry(self, scenario="normal"):
        self.state.update_telemetry(scenario=scenario)

    def update_node_packet_loss(self, node_id, packets_sent, packets_received, packet_loss_pct, semantic_meaning=None):
        self.state.update_node_packet_loss(node_id, packets_sent, packets_received, packet_loss_pct, semantic_meaning=semantic_meaning)

    def toggle_node(self, node_id):
        return self.state.toggle_node_state(node_id)

    def set_node_state(self, node_id, state):
        self.state.set_node_state(node_id, state)

    def get_virtual_summary(self):
        s_dict = self.state.get_summary()
        summary = self.topology.get_topology_summary()

        server_1_virtual = []
        for n in summary["server_1_nodes"]:
            n_id = n["node_id"]
            t_data = self.state.node_telemetry.get(n_id, {})
            server_1_virtual.append({
                "node_id": n_id,
                "virtual_name": f"Virtual Device {n_id}",
                "server_id": "Virtual Server 1",
                "status": t_data.get("status", "ACTIVE"),
                "snr_db": t_data.get("snr_db", 0.0),
                "latency_ms": t_data.get("latency_ms", 0.0),
                "bandwidth_mbps": t_data.get("bandwidth_mbps", 0.0),
                "packet_loss_pct": t_data.get("packet_loss_pct", 0.0),
                "packets_sent": t_data.get("packets_sent", 100),
                "packets_received": t_data.get("packets_received", 100),
                "packets_lost": t_data.get("packets_lost", 0),
                "delivery_status": t_data.get("delivery_status", "DELIVERED"),
                "semantic_meaning": t_data.get("semantic_meaning", "DOG"),
                "health": t_data.get("health", "UNKNOWN")
            })

        server_2_virtual = []
        for n in summary["server_2_nodes"]:
            n_id = n["node_id"]
            t_data = self.state.node_telemetry.get(n_id, {})
            server_2_virtual.append({
                "node_id": n_id,
                "virtual_name": f"Virtual Device {n_id}",
                "server_id": "Virtual Server 2",
                "status": t_data.get("status", "ACTIVE"),
                "snr_db": t_data.get("snr_db", 0.0),
                "latency_ms": t_data.get("latency_ms", 0.0),
                "bandwidth_mbps": t_data.get("bandwidth_mbps", 0.0),
                "packet_loss_pct": t_data.get("packet_loss_pct", 0.0),
                "packets_sent": t_data.get("packets_sent", 100),
                "packets_received": t_data.get("packets_received", 100),
                "packets_lost": t_data.get("packets_lost", 0),
                "delivery_status": t_data.get("delivery_status", "DELIVERED"),
                "semantic_meaning": t_data.get("semantic_meaning", "DOG"),
                "health": t_data.get("health", "UNKNOWN")
            })

        return {
            "virtual_servers": [
                {"server_id": "Virtual Server 1", "connected_virtual_nodes": ["N1", "N2", "N3", "N4", "N5"], "status": "ONLINE"},
                {"server_id": "Virtual Server 2", "connected_virtual_nodes": ["N6", "N7", "N8", "N9", "N10"], "status": "ONLINE"}
            ],
            "virtual_stats": s_dict,
            "server_1_virtual_nodes": server_1_virtual,
            "server_2_virtual_nodes": server_2_virtual
        }

    def generate_twin_plotly_figure(self, semantic_meaning="DOG", selected_node="N3"):
        """Generates interactive Digital Twin 6G Network Graph with live telemetry tooltips."""
        graph_data = self.topology.get_graph_elements(semantic_meaning=semantic_meaning, selected_node=selected_node)
        node_dict = {n["id"]: n for n in graph_data["nodes"]}

        if isinstance(selected_node, (list, tuple, set)):
            selected_set = set(selected_node)
        else:
            selected_set = {str(selected_node)}

        # Edge line vectors
        edge_x_root, edge_y_root = [], []
        edge_x_s1, edge_y_s1 = [], []
        edge_x_s2, edge_y_s2 = [], []

        for src, dst in graph_data["edges"]:
            n1 = node_dict[src]
            n2 = node_dict[dst]
            if src == "Root":
                edge_x_root.extend([n1["x"], n2["x"], None])
                edge_y_root.extend([n1["y"], n2["y"], None])
            elif src == "Server 1":
                edge_x_s1.extend([n1["x"], n2["x"], None])
                edge_y_s1.extend([n1["y"], n2["y"], None])
            elif src == "Server 2":
                edge_x_s2.extend([n1["x"], n2["x"], None])
                edge_y_s2.extend([n1["y"], n2["y"], None])

        trace_root_links = go.Scatter(
            x=edge_x_root, y=edge_y_root,
            line=dict(width=3, color="#FF2A6D", dash="solid"),
            hoverinfo="none", mode="lines",
            name="Virtual 6G Core Link"
        )

        trace_s1_links = go.Scatter(
            x=edge_x_s1, y=edge_y_s1,
            line=dict(width=2, color="#05D9E8", dash="dot"),
            hoverinfo="none", mode="lines",
            name="Virtual Server 1 Link"
        )

        trace_s2_links = go.Scatter(
            x=edge_x_s2, y=edge_y_s2,
            line=dict(width=2, color="#00FF9F", dash="dot"),
            hoverinfo="none", mode="lines",
            name="Virtual Server 2 Link"
        )

        root_nodes = [n for n in graph_data["nodes"] if n["type"] == "root"]
        server_nodes = [n for n in graph_data["nodes"] if n["type"] == "server"]
        device_nodes = [n for n in graph_data["nodes"] if n["type"] == "node"]

        trace_root = go.Scatter(
            x=[n.get("x", 0) for n in root_nodes],
            y=[n.get("y", 3) for n in root_nodes],
            mode="markers+text",
            hoverinfo="text",
            hovertext=[f"🔮 Virtual Semantic Core<br><b>Meaning: {semantic_meaning}</b>" for n in root_nodes],
            text=[f"Virtual Semantic: {semantic_meaning}" for n in root_nodes],
            textposition="top center",
            marker=dict(size=36, color="#FF2A6D", symbol="diamond", line=dict(width=3, color="#FFFFFF")),
            name="Virtual 6G Core"
        )

        s1_selected = any(n in selected_set for n in ["N1", "N2", "N3", "N4", "N5"])
        s2_selected = any(n in selected_set for n in ["N6", "N7", "N8", "N9", "N10"])

        trace_servers = go.Scatter(
            x=[n.get("x", 0) for n in server_nodes],
            y=[n.get("y", 1.8) for n in server_nodes],
            mode="markers+text",
            hoverinfo="text",
            hovertext=[f"🔮 Virtual {n.get('id', 'Server')}" for n in server_nodes],
            text=[f"Virtual {n.get('id', 'Server')}" for n in server_nodes],
            textposition="middle right",
            marker=dict(size=32, color=["#FFD700" if (n.get("id") == "Server 1" and s1_selected) or (n.get("id") == "Server 2" and s2_selected) else "#05D9E8" for n in server_nodes], symbol="square", line=dict(width=2, color="#FFFFFF")),
            name="Virtual Servers"
        )

        # Dynamic color coding and detailed live telemetry hovertext
        dev_colors = []
        dev_symbols = []
        dev_sizes = []
        dev_texts = []
        dev_labels = []
        for n in device_nodes:
            n_id = n.get("id", "N1")
            is_selected = (n_id in selected_set)
            t_data = self.state.node_telemetry.get(n_id, {})
            st_val = t_data.get("status", "ACTIVE")
            deliv_st = t_data.get("delivery_status", "DELIVERED" if st_val == "ACTIVE" else "FAILED")
            sem_txt = t_data.get("semantic_meaning", semantic_meaning).upper()

            if st_val == "INACTIVE":
                color = "#FF2A6D"  # Crimson Red for INACTIVE
                symbol = "x" if not is_selected else "star"
                size = 30 if is_selected else 24
                txt = f"❌ {n_id} [FAILED]" if not is_selected else f"❌ 🎯 {n_id} [FAILED]"
            else:
                color = "#FFD700" if is_selected else "#00FF9F"
                symbol = "star" if is_selected else "circle"
                size = 32 if is_selected else 24
                txt = f"🎯 {n_id} [{deliv_st}]" if is_selected else f"{n_id} [{deliv_st}]"

            dev_colors.append(color)
            dev_symbols.append(symbol)
            dev_sizes.append(size)
            dev_texts.append(txt)

            sel_hdr = f"🎯 <b>SELECTED DESTINATION NODE ({n_id})</b><br>" if is_selected else ""
            deliv_color = "#00FF9F" if deliv_st == "DELIVERED" else ("#FFD700" if deliv_st == "PARTIAL" else "#FF2A6D")
            
            if st_val == "ACTIVE":
                lbl = (
                    f"{sel_hdr}📱 <b>Virtual Device {n_id}</b><br>"
                    f"Connected Server: <b>{n.get('server', 'Server 1')}</b><br>"
                    f"Transmitted Semantic: <b style='color:#38BDF8;'>{sem_txt}</b><br>"
                    f"Delivery Status: <b style='color:{deliv_color};'>{deliv_st}</b><br>"
                    f"Packets: <b>{t_data.get('packets_sent', 100)} Sent / {t_data.get('packets_received', 100)} Recv / {t_data.get('packets_lost', 0)} Lost</b><br>"
                    f"Packet Loss: <b>{t_data.get('packet_loss_pct', 0.0)}%</b><br>"
                    f"SNR: <b>{t_data.get('snr_db', 0.0)} dB</b> | Latency: <b>{t_data.get('latency_ms', 0.0)} ms</b>"
                )
            else:
                lbl = (
                    f"{sel_hdr}📱 <b>Virtual Device {n_id}</b><br>"
                    f"Status: <b style='color:#FF2A6D;'>INACTIVE (OFFLINE)</b><br>"
                    f"Delivery Status: <b style='color:#FF2A6D;'>FAILED</b><br>"
                    f"Packet Loss: <b>100.0% (100/100 Lost)</b><br>"
                    f"SNR: <b>0.0 dB</b>"
                )
            dev_labels.append(lbl)

        trace_devices = go.Scatter(
            x=[n.get("x", 0) for n in device_nodes],
            y=[n.get("y", 0.2) for n in device_nodes],
            mode="markers+text",
            hoverinfo="text",
            hovertext=dev_labels,
            text=dev_texts,
            textposition="bottom center",
            textfont=dict(size=11, color="#FFFFFF"),
            marker=dict(size=dev_sizes, color=dev_colors, symbol=dev_symbols, line=dict(width=2, color="#FFFFFF")),
            name="Virtual Edge Devices"
        )

        fig = go.Figure(data=[
            trace_root_links, trace_s1_links, trace_s2_links,
            trace_root, trace_servers, trace_devices
        ])

        fig.add_shape(
            type="circle", xref="x", yref="y",
            x0=-5.0, y0=-0.3, x1=0.0, y1=2.3,
            fillcolor="rgba(5, 217, 232, 0.05)",
            line=dict(color="rgba(5, 217, 232, 0.3)", width=1.5, dash="dash")
        )
        fig.add_shape(
            type="circle", xref="x", yref="y",
            x0=0.0, y0=-0.3, x1=5.0, y1=2.3,
            fillcolor="rgba(0, 255, 159, 0.05)",
            line=dict(color="rgba(0, 255, 159, 0.3)", width=1.5, dash="dash")
        )

        sel_title_str = ", ".join(sorted(list(selected_set)))
        fig.add_annotation(
            x=-5.2, y=3.4,
            text=f"🔮 <b>Digital Twin Virtual Network Mirror (Selected Targets: {sel_title_str})</b>",
            showarrow=False,
            xanchor="left",
            font=dict(size=14, color="#05D9E8")
        )

        fig.update_layout(
            template="plotly_dark",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
            hovermode="closest",
            margin=dict(b=40, l=40, r=40, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5.5, 5.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.8, 3.8]),
            height=500,
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117"
        )

        return fig

    def generate_telemetry_bar_chart(self, selected_node="N3"):
        """Generates Plotly comparison bar charts for SNR (dB) and Latency (ms) across N1..N10."""
        telemetry_list = list(self.state.node_telemetry.values())
        nodes = [t["node_id"] for t in telemetry_list]
        snrs = [t["snr_db"] for t in telemetry_list]
        lats = [t["latency_ms"] for t in telemetry_list]

        if isinstance(selected_node, (list, tuple, set)):
            selected_set = set(selected_node)
        else:
            selected_set = {str(selected_node)}

        snr_colors = ["#FFD700" if n in selected_set else "#00FF9F" for n in nodes]
        lat_colors = ["#FF8C00" if n in selected_set else "#FF2A6D" for n in nodes]

        sel_title_str = ", ".join(sorted(list(selected_set)))
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=nodes,
            y=snrs,
            name="SNR (dB)",
            marker_color=snr_colors
        ))
        fig.add_trace(go.Bar(
            x=nodes,
            y=lats,
            name="Latency (ms)",
            marker_color=lat_colors
        ))
        fig.update_layout(
            title=f"📡 Network Telemetry Comparison: SNR (dB) vs. Latency (ms) [Target Nodes: {sel_title_str}]",
            barmode="group",
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117"
        )
        return fig


if __name__ == "__main__":
    twin = DigitalTwinEngine()
    summary = twin.get_virtual_summary()
    print("============================================================")
    print(" PHASE 7: DIGITAL TWIN WITH TELEMETRY INITIALIZED")
    print("============================================================")
    print(f" Avg SNR: {summary['virtual_stats']['avg_snr_db']} dB")
    print(f" Avg Latency: {summary['virtual_stats']['avg_latency_ms']} ms")
    print(f" Avg Bandwidth: {summary['virtual_stats']['avg_bandwidth_mbps']} Mbps")
    print(f" Avg Packet Loss: {summary['virtual_stats']['avg_packet_loss_pct']}%")
    print("============================================================")
