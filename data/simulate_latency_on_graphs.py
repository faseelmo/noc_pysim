import os
from natsort import natsorted
import networkx as nx

from src.utils import graph_to_task_list
from src.utils import get_random_packet_list
from src.utils import simulate

from data.utils import load_graph_from_json, visualize_graph


def simlate_latency_from_graph(nx_graph: nx.DiGraph, debug_mode: bool):
    computing_list = graph_to_task_list(nx_graph)
    packet_list = get_random_packet_list(nx_graph)
    latency = simulate(computing_list, packet_list, debug_mode=debug_mode)
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
        "--sim",
        action="store_true",
        help="Get latency of all the graphs in data/pe_task_graphs/",
    )
    args = parser.parse_args()

    if args.test:
        # graph = load_graph_from_json("data/test_task_graph.json")
        graph = load_graph_from_json("data/pe_task_graphs/task_graph_0.json")
        visualize_graph(graph)
        latency = simlate_latency_from_graph(graph, debug_mode=True)
        print(f"\nLatency of the test graph in /data/test_task_graph.json is {latency}")

    if args.sim:
        list_of_files = [
            f for f in os.listdir("data/pe_task_graphs/") if f.endswith(".json")
        ]
        list_of_files = natsorted(list_of_files)

        for file in list_of_files:

            graph = load_graph_from_json(f"data/pe_task_graphs/{file}")
            latency = simlate_latency_from_graph(graph, debug_mode=False)
            if latency == 999:
                print("Graph is not schedulable")

                break
            print(f"Latency of the graph {file} is {latency}")
