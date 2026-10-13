class NetworkNode:
    """
    Simulated 6G Edge Node in Network Topology.
    N1..N5 attached to Server 1, N6..N10 attached to Server 2.
    """
    def __init__(self, node_id, server_id, name=None):
        self.node_id = node_id
        self.name = name or node_id
        self.server_id = server_id
        self.status = "ONLINE"

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "name": self.name,
            "server_id": self.server_id,
            "status": self.status
        }

    def __repr__(self):
        return f"<Node {self.node_id} -> {self.server_id}>"
