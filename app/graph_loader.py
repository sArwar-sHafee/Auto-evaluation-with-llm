import json
def load_graph_data(graph_data_path: str) -> dict:
    """
    Load the graph data from the given path.
    """
    with open(graph_data_path, "r") as f:
        graph_data = json.load(f)
    return graph_data