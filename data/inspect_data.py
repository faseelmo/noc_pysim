import os 
import json 
from natsort import natsorted

from data.utils import load_graph_from_json, visualize_graph

if __name__ == "__main__":

    input_path = os.path.join("data","training_data", "input")
    target_path = os.path.join("data","training_data", "target")

    input_files = natsorted(os.listdir(input_path))
    target_files = natsorted(os.listdir(target_path))

    for input, target in zip(input_files, target_files):
        graph = load_graph_from_json(os.path.join(input_path, input))
        latency = json.load(open(os.path.join(target_path, target)))["latency"]
        visualize_graph(graph,latency )
