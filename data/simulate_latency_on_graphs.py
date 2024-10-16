import os
import json
import random
import networkx as nx
from natsort import natsorted

from src.utils import simulate
from src.utils import graph_to_task_list
from src.utils import get_random_packet_list, get_ordered_packet_list

from data.utils import (
    load_graph_from_json, 
    visualize_graph, 
    does_path_contains_files, 
    compute_list_to_node_dict
)

def simlate_latency_from_graph(nx_graph: nx.DiGraph, debug_mode: bool, max_cycles: int, sjf_scheduling: bool):
    random.seed(0)
    computing_list  = graph_to_task_list(nx_graph)
    packet_list     = get_ordered_packet_list(nx_graph)
    # packet_list     = get_random_packet_list(nx_graph)

    packet_list_copy = []
    for packet in packet_list:
        packet_list_copy.append(packet.get_source_task_id())

    if debug_mode:
        print(f"---- Debug Mode ----")
        print(f"\nPacket List")
        print(*packet_list)

        print(f"\nComputing List")
        for idx, task in enumerate(computing_list):
            print(f"{idx}. {task}\n")


        visualize_graph(nx_graph)
        print(f"\n")

    latency     = simulate(
        computing_list, 
        packet_list, 
        debug_mode      = debug_mode, 
        max_cycles      = max_cycles,
        sjf_scheduling  = sjf_scheduling
    )

    return latency, packet_list_copy, computing_list


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="Get latency of test graph in /data/test_task_graph.json",
    )
    parser.add_argument(
        "--analytical",
        action="store_true",
        help="Get latency of graph in /data/analytical_test_data/input/index.json, default index is 1",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Index of the graph in /data/analytical_test_data/input/index.json",
        default=1,
    )
    parser.add_argument(
        "--max_cycle",
        type=int,
        help="Max cycles to run the simulation for in test mode. Default is 1000",
        default=1000,
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Get latency of all the graphs in data/pe_task_graphs/",
    )
    parser.add_argument(
        "--sjf",
        action="store_true",
        help="Use Shortest Job First scheduling"
    )
    args = parser.parse_args()

    if not args.test and not args.sim and not args.analytical:
        print("Please specify --test, --analytical or --sim")
        exit()

    if args.test:
        # graph_path  = "data/test_task_graph.json"
        graph_path  = "data/training_data/input/task_graph_0.json"

        graph       = load_graph_from_json(graph_path)

        latency, packet_list, computing_list = simlate_latency_from_graph(
            graph, debug_mode=True, max_cycles=args.max_cycle, sjf_scheduling=args.sjf
        )

        for task in computing_list:
            print(f"Task {task.task_id} starts at {task.start_cycle} and ends at {task.end_cycle}")

        visualize_graph(graph, latency_value=latency, compute_list=computing_list, packet_list=packet_list)

        print(f"\nLatency of the test graph in {graph_path} is {latency}")

    if args.analytical:
        analytical_idx = args.index
        graph_path      = f"data/analytical_test_data/input/{analytical_idx}.json"

        graph       = load_graph_from_json(graph_path)

        latency, packet_list, computing_list = simlate_latency_from_graph(
            graph, debug_mode=True, max_cycles=args.max_cycle, sjf_scheduling=args.sjf
        )

        with open(f"data/analytical_test_data/target/{analytical_idx}.json", "w") as f:
            node_time_info = compute_list_to_node_dict(computing_list)
            target_json = {"latency": latency}
            target_json.update(node_time_info)
            f.write(json.dumps(target_json))

        visualize_graph(graph, latency_value=latency, compute_list=computing_list, packet_list=packet_list)

    if args.sim:

        INPUT_DATA_DIR  = "data/training_data/input"
        TARGET_DATA_DIR = "data/training_data/target"
        PACKET_LIST_DIR = "data/training_data/packet_list"

        print(f"\nCreating Latency data (using noc_pysim) in {TARGET_DATA_DIR}")
        does_path_contains_files(TARGET_DATA_DIR)

        list_of_files = [f for f in os.listdir(INPUT_DATA_DIR) if f.endswith(".json")]
        list_of_files = natsorted(list_of_files)

        for file in list_of_files:

            graph = load_graph_from_json(f"{INPUT_DATA_DIR}/{file}")

            latency, packet_list, computing_list = simlate_latency_from_graph(
                graph, debug_mode=False, max_cycles=args.max_cycle, sjf_scheduling=args.sjf
            )

            assert (
                latency != 999 or latency != 0
            ), f"Graph task {file} is not schedulable. Latency is {latency}"

            node_time_info = compute_list_to_node_dict(computing_list)
            latency_json = {"latency": latency}
            latency_json.update(node_time_info)

            packet_list_json = json.dumps(packet_list)

            with open(f"{TARGET_DATA_DIR}/{file}", "w") as f:
                f.write(json.dumps(latency_json))

            with open(f"{PACKET_LIST_DIR}/{file}", "w") as f:
                f.write(packet_list_json)

            print(f"Latency of the graph {file} is {latency}")
