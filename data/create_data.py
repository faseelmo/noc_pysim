import networkx as nx
import random
import json


def generate_graph(num_nodes: int):
    """
    Generates a GNR (Growing network with reduction)
    or GNC (Growing network with copying)
    Graph will have arg "num_nodes" number of nodes
    """
    redirection_probability = random.uniform(0, 0.3)
    graph_generator = [
        lambda: nx.gnr_graph(num_nodes, redirection_probability),
        lambda: nx.gnc_graph(num_nodes),
    ]
    return random.choice(graph_generator)()


def modify_graph_to_task_graph(graph: nx.DiGraph):
    """
    Add Task information as node and edge attributes to the arg "graph"
    """
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


def save_graph_to_json(graph: nx.DiGraph, filename: str):
    data = nx.node_link_data(graph)
    with open(filename, "w") as file:
        json.dump(data, file)


def load_graph_from_json(filename: str):
    with open(filename, "r") as file:
        data = json.load(file)
    return nx.node_link_graph(data)


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
        node: f"{node}\nG: {graph.nodes[node].get('generate', 'N/A')}"
        for node in graph.nodes
    }
    nx.draw_networkx_labels(graph, pos, labels=custom_labels)

    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.show()

def test_function(num_nodes: int):
    graph = generate_graph(num_nodes)
    modified_graph = modify_graph_to_task_graph(graph)
    visualize_modified_graph(modified_graph)

    save_graph_to_json(modified_graph, "test_task_graph.json")
    loaded_graph = load_graph_from_json("test_task_graph.json")
    visualize_modified_graph(loaded_graph)

def generate_n_graphs(count: int): 
    for i in range(count):
        random_num_nodes = random.randint(2, 6)
        graph = generate_graph(random_num_nodes)
        modified_graph = modify_graph_to_task_graph(graph)

        save_graph_to_json(modified_graph, f"pe_task_graphs/task_graph_{i}.json")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=4,
        help="Number of nodes in the generated graph",
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Run the test function"
    )
    parser.add_argument(
        "--generate", 
        action="store_true",
        help="Generate"
    )
    parser.add_argument(
        "--gen_count", 
        type=int,
        default=100,
        help="Number of graphs to generate"
    )

    args = parser.parse_args()
    print(f"Number of nodes is {args.num_nodes}, testing is {args.test}, generating is {args.generate}, gen_count is {args.gen_count}")

    if args.test:
        test_function(args.num_nodes)

    if args.generate:
        generate_n_graphs(args.gen_count)

