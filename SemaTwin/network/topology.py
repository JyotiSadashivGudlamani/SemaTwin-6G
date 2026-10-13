import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from network.server import ContentServer
from network.node import NetworkNode


class NetworkTopology:
    """
    Phase 5: Network Topology Module.
    Basic 6G Network Topology containing:
    - 2 Edge/Content Servers ("Server 1", "Server 2")
    - Server 1 connected ONLY to 5 mobile/edge nodes: N1, N2, N3, N4, N5
    - Server 2 connected ONLY to 5 mobile/edge nodes: N6, N7, N8, N9, N10
    """
    def __init__(self):
        self.servers = {
            "Server 1": ContentServer("Server 1", name="Edge Content Server 1"),
            "Server 2": ContentServer("Server 2", name="Edge Content Server 2")
        }
        self.nodes = {}
        self._build_topology()

    def _build_topology(self):
        # Server 1 -> Mobile/Edge Devices N1..N5
        for i in range(1, 6):
            n_id = f"N{i}"
            node = NetworkNode(node_id=n_id, server_id="Server 1", name=f"Edge Device {n_id}")
            self.nodes[n_id] = node
            self.servers["Server 1"].register_node(n_id)

        # Server 2 -> Mobile/Edge Devices N6..N10
        for i in range(6, 11):
            n_id = f"N{i}"
            node = NetworkNode(node_id=n_id, server_id="Server 2", name=f"Edge Device {n_id}")
            self.nodes[n_id] = node
            self.servers["Server 2"].register_node(n_id)

    def update_all_telemetry(self, scenario="normal"):
        """Updates telemetry for all nodes in the topology."""
        return True

    def get_topology_summary(self):
        return {
            "servers": [s.to_dict() for s in self.servers.values()],
            "total_servers": len(self.servers),
            "total_nodes": len(self.nodes),
            "server_1_nodes": [n.to_dict() for n in self.nodes.values() if n.server_id == "Server 1"],
            "server_2_nodes": [n.to_dict() for n in self.nodes.values() if n.server_id == "Server 2"]
        }

    def get_graph_elements(self, semantic_meaning="DOG", selected_node="N3"):
        """
        Returns futuristic 6G Network Topology graph structure:
        Semantic Meaning (6G Core) -> Server 1 / Server 2 -> Mobile Edge Devices (N1-N5 / N6-N10)
        selected_node can be a single node ID ("N3") or a list/set of node IDs (["N1", "N2", "N3"]).
        """
        nodes = []
        edges = []

        if isinstance(selected_node, (list, tuple, set)):
            selected_set = set(selected_node)
        else:
            selected_set = {str(selected_node)}

        # Root Node: 6G Semantic AI Core
        root_label = f"🧠 6G Semantic Core<br>(Meaning: {semantic_meaning})"
        nodes.append({
            "id": "Root",
            "label": root_label,
            "name": f"Semantic: {semantic_meaning}",
            "type": "root",
            "x": 0.0,
            "y": 3.0,
            "color": "#FF2A6D",  # Neon Crimson
            "size": 36,
            "symbol": "diamond"
        })

        s1_selected = any(n in selected_set for n in ["N1", "N2", "N3", "N4", "N5"])
        s2_selected = any(n in selected_set for n in ["N6", "N7", "N8", "N9", "N10"])

        # Edge Content Servers
        nodes.append({
            "id": "Server 1",
            "label": "🖥️ Edge Server 1 (ACTIVE SOURCE)" if s1_selected else "🖥️ Edge Server 1",
            "name": "Server 1",
            "type": "server",
            "x": -2.5,
            "y": 1.8,
            "color": "#FFD700" if s1_selected else "#05D9E8",
            "size": 34 if s1_selected else 28,
            "symbol": "square"
        })

        nodes.append({
            "id": "Server 2",
            "label": "🖥️ Edge Server 2 (ACTIVE SOURCE)" if s2_selected else "🖥️ Edge Server 2",
            "name": "Server 2",
            "type": "server",
            "x": 2.5,
            "y": 1.8,
            "color": "#FFD700" if s2_selected else "#05D9E8",
            "size": 34 if s2_selected else 28,
            "symbol": "square"
        })

        edges.append(("Root", "Server 1"))
        edges.append(("Root", "Server 2"))

        # Server 1 Mobile Edge Devices (N1..N5)
        for idx, n_id in enumerate(["N1", "N2", "N3", "N4", "N5"]):
            x_pos = -4.5 + idx * 1.0
            is_selected = (n_id in selected_set)
            nodes.append({
                "id": n_id,
                "label": f"🎯 Edge Device {n_id} (SELECTED)" if is_selected else f"📱 Edge Device {n_id}",
                "name": f"🎯 {n_id}" if is_selected else f"Device {n_id}",
                "short_label": f"🎯 {n_id}" if is_selected else n_id,
                "type": "node",
                "server": "Server 1",
                "x": x_pos,
                "y": 0.2,
                "color": "#FFD700" if is_selected else "#00FF9F",  # Gold for selected, Emerald for normal
                "size": 32 if is_selected else 22,
                "symbol": "star" if is_selected else "circle",
                "is_selected": is_selected
            })
            edges.append(("Server 1", n_id))

        # Server 2 Mobile Edge Devices (N6..N10)
        for idx, n_id in enumerate(["N6", "N7", "N8", "N9", "N10"]):
            x_pos = 0.5 + idx * 1.0
            is_selected = (n_id in selected_set)
            nodes.append({
                "id": n_id,
                "label": f"🎯 Edge Device {n_id} (SELECTED)" if is_selected else f"📱 Edge Device {n_id}",
                "name": f"🎯 {n_id}" if is_selected else f"Device {n_id}",
                "short_label": f"🎯 {n_id}" if is_selected else n_id,
                "type": "node",
                "server": "Server 2",
                "x": x_pos,
                "y": 0.2,
                "color": "#FFD700" if is_selected else "#00FF9F",  # Gold for selected, Emerald for normal
                "size": 32 if is_selected else 22,
                "symbol": "star" if is_selected else "circle",
                "is_selected": is_selected
            })
            edges.append(("Server 2", n_id))

        return {"nodes": nodes, "edges": edges}


if __name__ == "__main__":
    topo = NetworkTopology()
    summary = topo.get_topology_summary()
    print("============================================================")
    print(" PHASE 5: FUTURISTIC 6G NETWORK TOPOLOGY INITIALIZED")
    print("============================================================")
    print(f" Total Servers: {summary['total_servers']}")
    print(f" Total Nodes:   {summary['total_nodes']}")
    print(f" Server 1 Nodes: {[n['node_id'] for n in summary['server_1_nodes']]}")
    print(f" Server 2 Nodes: {[n['node_id'] for n in summary['server_2_nodes']]}")
    print("============================================================")
