import networkx as nx
import random


def generate_graph(num_nodes: int):

    redirection_probability = random.uniform(0, 0.3)
    graph_generator = [
        lambda: nx.gnr_graph(num_nodes, redirection_probability),
        lambda: nx.gnc_graph(num_nodes),
    ]
    return random.choice(graph_generator)()


def modify_graph_to_task_graph(graph: nx.DiGraph):
    require_generate_range = (1, 10)

    for node in graph.nodes:
        successors = list(graph.successors(node))
        predecessors = list(graph.predecessors(node))
        generate = random.randint(*require_generate_range)
        processing_time = random.randint(*require_generate_range)

        # If the node has no incoming edges
        if len(predecessors) == 0:
            graph.nodes[node]["type"] = "dependency"
            for successor in successors:
                require = random.randint(*require_generate_range)
                graph.add_edge(node, successor, weight=require)

        # If node has no outgoing edges
        elif len(successors) == 0:
            graph.nodes[node].update(
                {
                    "type": "task",
                    "generate": generate,
                    "processing_time": processing_time,
                }
            )

        # If node has both incoming and outgoing edges
        elif len(successors) > 0 and len(predecessors) > 0:
            graph.nodes[node].update(
                {
                    "type": "task",
                    "generate": generate,
                    "processing_time": processing_time,
                }
            )
            for successor in successors:
                require = random.randint(*require_generate_range)
                graph.add_edge(node, successor, weight=require)
            for predecessor in predecessors:
                require = random.randint(*require_generate_range)
                graph.add_edge(predecessor, node, weight=require)

        else:
            raise ValueError("Dangling node detected")

    return graph


def visualize_modified_graph(graph: nx.DiGraph):
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
        node_size=700,
        node_color=node_colors,
        arrows=True,
    )

    custom_labels = {
        node: f"{node}\n(G: {graph.nodes[node].get('generate', 'N/A')})"
        for node in graph.nodes
    }
    nx.draw_networkx_labels(graph, pos, labels=custom_labels)

    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.show()


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # num of nodes range from 2 to 6
    number_of_nodes = 2
    graph = generate_graph(number_of_nodes)
    modified_graph = modify_graph_to_task_graph(graph)
    visualize_modified_graph(modified_graph)
