import os
import json
import argparse

from natsort import natsorted

from data.utils import load_graph_from_json, visualize_graph


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test mode")
    args = parser.parse_args()

    if args.test:
        input_path = os.path.join("data", "training_data", "test", "input")
        target_path = os.path.join("data", "training_data", "test", "target")

    else:
        input_path = os.path.join("data", "training_data", "input")
        target_path = os.path.join("data", "training_data", "target")

    input_files = natsorted(os.listdir(input_path))
    target_files = natsorted(os.listdir(target_path))

    for input, target in zip(input_files, target_files):
        graph = load_graph_from_json(os.path.join(input_path, input))
        latency = json.load(open(os.path.join(target_path, target)))["latency"]
        visualize_graph(graph, latency)
