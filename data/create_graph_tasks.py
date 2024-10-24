import random
import networkx as nx
import numpy as np
from data.utils import (
    save_graph_to_json,
    load_graph_from_json,
    visualize_graph,
    does_path_contains_files,
)

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
    max_generate = 10
    processing_time_range = (1, 10)

    for node in graph.nodes:
        successors          = list(graph.successors(node))
        predecessors        = list(graph.predecessors(node))
        num_of_successors   = len(successors)

        generate_range          = (num_of_successors + 1, max_generate)
        random_generate_value   = random.randint(*generate_range)

        gen_split_values = get_split_value(random_generate_value, num_of_successors)

        # Condition: If the node has no incoming edges (dependency node)
        # Note: dependency nodes don't have processing time
        if len(predecessors) == 0:

            graph.nodes[node]["type"]               = "dependency"
            graph.nodes[node]["generate"]           = random_generate_value
            graph.nodes[node]["processing_time"]    = 0
            graph.nodes[node]["wait_time"]          = 0

            for successor, gen_value in zip(successors, gen_split_values):
                graph[node][successor]["weight"] = gen_value

        else:

            random_processing_time = random.randint(*processing_time_range)

            graph.nodes[node]["type"]               = "task"
            graph.nodes[node]["generate"]           = random_generate_value
            graph.nodes[node]["processing_time"]    = random_processing_time
            graph.nodes[node]["wait_time"]          = 0

            # Assigning require (edge weights) to successors by
            # splitting the generate value randomly

            gen_split_values = get_split_value(random_generate_value, num_of_successors)

            for successor, gen_value in zip(successors, gen_split_values):

                require = gen_value
                graph[node][successor]["weight"] = int(require)

        if len(predecessors) == 0 and len(successors) == 0:
            raise ValueError("Dangling node detected")

    for node in graph.nodes:
        # - Changing nodes that have 'dependency' predecessors to 'task_depend',
        #   from type 'task' to 'task_depend'
        # - Changing the wait time of the task_depend node to 
        #   max( 4 * require_value ) of its predecessors. 
        #  Note: 4 is the packet size in flit

        if graph.nodes[node]["type"] != "dependency":
            continue

        successors = list(graph.successors(node))
        max_weight = max([graph[node][successor]["weight"] for successor in successors])

        graph.nodes[node]["generate"] = max_weight

        for successor in successors:
            
            graph.nodes[successor]["type"] = "task_depend"

            # require_value   = graph[node][successor]["weight"]
            
            predecessors    = list(graph.predecessors(successor))
            require_value   = sum([graph[predecessor][successor]["weight"] for predecessor in predecessors])

            wait_time       = 4 * require_value # 4 is the packet size in flit

            current_node_wait_time = graph.nodes[successor]["wait_time"]

            if wait_time > current_node_wait_time:
                graph.nodes[successor]["wait_time"] = wait_time

    return graph


def get_split_value(generate_value: int, num_of_successors: int):
    assert (
        generate_value >= num_of_successors
    ), "generate_value must be at least as large as num_of_successors"

    base_values     = np.ones(num_of_successors, dtype=int)  # assign 1 to each successor
    remaining_value = generate_value - num_of_successors

    additional_values = np.random.multinomial(
        remaining_value, np.ones(num_of_successors) / num_of_successors
    )

    gen_split_values = base_values + additional_values
    gen_split_values = (
        gen_split_values.tolist()
    )  # Converts to list for json serialization

    for value in gen_split_values:
        assert value > 0, f"Generate split value is {value}"

    return gen_split_values


def test_function(num_nodes: int):
    graph           = generate_graph(num_nodes)
    modified_graph  = modify_graph_to_task_graph(graph)
    visualize_graph(modified_graph)

    save_graph_to_json(modified_graph, "data/test_task_graph.json")
    loaded_graph    = load_graph_from_json("data/test_task_graph.json")
    visualize_graph(loaded_graph)


def generate_n_graphs(count: int, num_nodes: int):
    path = "data/training_data/input"

    print(f"\nCreating Graphs in {path}")
    does_path_contains_files(path)

    for i in range(count):
        random_num_nodes    = random.randint(2, num_nodes)
        graph               = generate_graph(random_num_nodes)
        modified_graph      = modify_graph_to_task_graph(graph)

        save_graph_to_json(
            modified_graph, f"data/training_data/input/task_graph_{i}.json"
        )

    print(f"{count} graphs generated")


if __name__ == "__main__":

    import argparse

    random.seed(11)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="Generates one graph and stores it in data/test_task_graph.json",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=4,
        help='Number of nodes in the "test" generated graph. Or max number of nodes in the automated generated graphs',
    )
    parser.add_argument("--generate", action="store_true", help="Generate flag")
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
        generate_n_graphs(args.gen_count, args.num_nodes)
