class SemanticRouter:
    """
    Semantic Data Router.
    Routes semantic packets from Content Server 1 or 2 to target edge nodes.
    """
    def __init__(self, topology):
        self.topology = topology

    def route_packet(self, packet, target_node_id):
        if target_node_id not in self.topology.nodes:
            return {
                "success": False,
                "error": f"Node {target_node_id} does not exist in network topology."
            }

        target_node = self.topology.nodes[target_node_id]
        parent_server_id = target_node.server_id
        server = self.topology.servers[parent_server_id]

        # Route through server
        route_res = server.route_semantic_packet(packet, target_node_id)
        rx_res = target_node.receive_semantic_packet(packet)

        return {
            "success": True,
            "packet_id": packet.packet_id,
            "source_server": server.name,
            "target_node": target_node_id,
            "path": f"{server.name} -> {target_node_id} -> Receiver",
            "semantic_class": packet.predicted_class,
            "node_snr_db": target_node.snr_db,
            "node_latency_ms": target_node.latency_ms,
            "node_status": target_node.status,
            "transmission_status": "Semantic data received successfully."
        }
