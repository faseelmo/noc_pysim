import os
import json
import random
import networkx as nx
from natsort import natsorted

from src.utils import simulate
from src.utils import graph_to_task_list
from src.utils import get_random_packet_list

from data.utils import load_graph_from_json, visualize_graph


def simlate_latency_from_graph(nx_graph: nx.DiGraph, debug_mode: bool, max_cycles: int):
    random.seed(0)
    computing_list = graph_to_task_list(nx_graph)
    packet_list = get_random_packet_list(nx_graph)
    if debug_mode:
        print(*packet_list)
    latency = simulate(
        computing_list, packet_list, debug_mode=debug_mode, max_cycles=max_cycles
    )
    return latency


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="Get latency of test graph in /data/test_task_graph.json",
    )
    parser.add_argument(
        "--max_cycle",
        type=int,
        help="Max cycles to run the simulation for in test mode",
        default=1000,
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Get latency of all the graphs in data/pe_task_graphs/",
    )
    args = parser.parse_args()

    if not args.test and not args.sim:
        print("Please specify --test or --sim")
        exit()

    if args.test:
        # graph = load_graph_from_json("data/test_task_graph.json")
        graph_path = "data/training_data/input/task_graph_0.json"
        graph = load_graph_from_json(graph_path)
        visualize_graph(graph)
        latency = simlate_latency_from_graph(
            graph, debug_mode=True, max_cycles=args.max_cycle
        )
        print(f"\nLatency of the test graph in {graph_path} is {latency}")

    if args.sim:

        INPUT_DATA_DIR = "data/training_data/input"
        TARGET_DATA_DIR = "data/training_data/target"

        list_of_files = [f for f in os.listdir(INPUT_DATA_DIR) if f.endswith(".json")]
        list_of_files = natsorted(list_of_files)

        for file in list_of_files:

            graph = load_graph_from_json(f"{INPUT_DATA_DIR}/{file}")
            latency = simlate_latency_from_graph(
                graph, debug_mode=False, max_cycles=1000
            )

            assert (
                latency != 999 or latency != 0
            ), f"Graph task {file} is not schedulable. Latency is {latency}"

            latency_json = {"latency": latency}

            with open(f"{TARGET_DATA_DIR}/{file}", "w") as f:
                f.write(json.dumps(latency_json))

            print(f"Latency of the graph {file} is {latency}")
