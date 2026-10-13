class ContentServer:
    """
    Simulated Content Server in 6G Network Topology.
    - Server 1 ("Server 1") connected to N1, N2, N3, N4, N5
    - Server 2 ("Server 2") connected to N6, N7, N8, N9, N10
    """
    def __init__(self, server_id, name="Server 1"):
        self.server_id = server_id
        self.name = name
        self.connected_nodes = []
        self.status = "ONLINE"

    def register_node(self, node_id):
        if node_id not in self.connected_nodes:
            self.connected_nodes.append(node_id)

    def to_dict(self):
        return {
            "server_id": self.server_id,
            "name": self.name,
            "status": self.status,
            "connected_nodes": self.connected_nodes
        }

    def __repr__(self):
        return f"<Server {self.server_id}: {self.name} | Nodes: {self.connected_nodes}>"
