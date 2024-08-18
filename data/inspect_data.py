import os
import json
import argparse

from natsort import natsorted

from data.utils import load_graph_from_json, visualize_graph, get_compute_list_from_json


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test mode")
    args = parser.parse_args()

    if args.test:
        input_path = os.path.join("data", "training_data", "test", "input")
        target_path = os.path.join("data", "training_data", "test", "target")
        packet_list_path = os.path.join("data", "training_data", "test", "packet_list")

    else:
        input_path = os.path.join("data", "training_data", "input")
        target_path = os.path.join("data", "training_data", "target")
        packet_list_path = os.path.join("data", "training_data", "packet_list")

    input_files = natsorted(os.listdir(input_path))
    target_files = natsorted(os.listdir(target_path))
    packet_list_files = natsorted(os.listdir(packet_list_path))

    for input_idx, target_idx, packet_list_idx in zip(input_files, target_files, packet_list_files):
        graph = load_graph_from_json(os.path.join(input_path, input_idx))
        
        json_data = json.load(open(os.path.join(target_path, target_idx)))

        latency = json_data["latency"]
        compute_list = get_compute_list_from_json(os.path.join(target_path, target_idx))

        packet_list = json.load(open(os.path.join(packet_list_path, packet_list_idx)))
        visualize_graph(graph, latency, packet_list, compute_list)