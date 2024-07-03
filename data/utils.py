import networkx as nx
import json


def save_graph_to_json(graph: nx.DiGraph, filename: str):
    data = nx.node_link_data(graph)
    with open(filename, "w") as file:
        json.dump(data, file)


def load_graph_from_json(filename: str):
    with open(filename, "r") as file:
        data = json.load(file)
    return nx.node_link_graph(data)


def visualize_graph(graph: nx.DiGraph):
    import matplotlib.pyplot as plt
    import networkx as nx

    color_map = {"dependency": "skyblue", "task": "lightgreen"}
    node_colors = [
        color_map.get(graph.nodes[node].get("type", "task"), "lightgreen")
        for node in graph.nodes
    ]

    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_size=900,
        node_color=node_colors,
        arrows=True,
    )

    custom_labels = {}
    for node in graph.nodes:
        label_parts = [f"id: {node}"]
        if "processing_time" in graph.nodes[node]:
            label_parts.append(f"P: {graph.nodes[node]['processing_time']}")
        if "generate" in graph.nodes[node]:
            label_parts.append(f"G: {graph.nodes[node]['generate']}")
        custom_labels[node] = "\n".join(label_parts)


    nx.draw_networkx_labels(graph, pos, labels=custom_labels)

    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.show()
