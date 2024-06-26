import random
import networkx as nx
import numpy as np
from data.utils import save_graph_to_json, load_graph_from_json, visualize_graph


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
    Add Task information as node (generate) and edge attributes (require)
    to the arg "graph"
    """
    generate_range = (1, 10)

    for node in graph.nodes:
        successors = list(graph.successors(node))
        predecessors = list(graph.predecessors(node))

        # Condition: If the node has no incoming edges (dependency node)
        # Note: dependency nodes don't have processing time
        if len(predecessors) == 0:
            random_require_value = random.randint(*generate_range)
            graph.nodes[node]["type"] = "dependency"

            for successor in successors:
                require = random_require_value

                graph[node][successor]["weight"] = require
                graph.nodes[node]["generate"] = require

        else:
            random_generate_value = random.randint(*generate_range)
            random_processing_time = random.randint(*generate_range)

            graph.nodes[node]["type"] = "task"
            graph.nodes[node]["generate"] = random_generate_value
            graph.nodes[node]["processing_time"] = random_processing_time

            # Assigning require (edge weights) to successors by
            # splitting the generate value randomly
            num_of_successors = len(successors)
            gen_split_values = np.random.multinomial(
                random_generate_value,
                np.ones(num_of_successors) / num_of_successors,
                size=1,
            )[0]

            for successor, gen_value in zip(successors, gen_split_values):

                require = gen_value
                graph[node][successor]["weight"] = int(require)

        if len(predecessors) == 0 and len(successors) == 0:
            raise ValueError("Dangling node detected")

    return graph


def test_function(num_nodes: int):
    graph = generate_graph(num_nodes)
    modified_graph = modify_graph_to_task_graph(graph)
    visualize_graph(modified_graph)

    save_graph_to_json(modified_graph, "data/test_task_graph.json")
    loaded_graph = load_graph_from_json("data/test_task_graph.json")
    visualize_graph(loaded_graph)


def generate_n_graphs(count: int):
    for i in range(count):
        random_num_nodes = random.randint(2, 6)
        graph = generate_graph(random_num_nodes)
        modified_graph = modify_graph_to_task_graph(graph)

        save_graph_to_json(modified_graph, f"data/pe_task_graphs/task_graph_{i}.json")


if __name__ == "__main__":

    import argparse

    # random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run the test function")
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=4,
        help="Number of nodes in the test generated graph",
    )
    parser.add_argument("--generate", action="store_true", help="Generate")
    parser.add_argument(
        "--gen_count", type=int, default=100, help="Number of graphs to generate"
    )

    args = parser.parse_args()
    print(
        f"Number of nodes is {args.num_nodes}, testing is {args.test}, generating is {args.generate}, gen_count is {args.gen_count}"
    )

    if args.test:
        test_function(args.num_nodes)

    if args.generate:
        generate_n_graphs(args.gen_count)
