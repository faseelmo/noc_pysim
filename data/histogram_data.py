from data.utils import load_graph_from_json

import os
from natsort import natsorted
import matplotlib.pyplot as plt


if __name__ == "__main__":

    training_data_path = "data/training_data/input"
    test_data_path = "data/training_data/test/input"

    training_input_files = natsorted(os.listdir(training_data_path))
    test_files = natsorted(os.listdir(test_data_path))

    training_node_counts = []
    for file in training_input_files:
        graph = load_graph_from_json(os.path.join(training_data_path, file))
        training_node_counts.append(len(graph.nodes))

    test_node_counts = []
    for file in test_files:
        graph = load_graph_from_json(os.path.join(test_data_path, file))
        test_node_counts.append(len(graph.nodes))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.hist(training_node_counts, bins=10, edgecolor="black")
    ax1.set_title("Training Data: Histogram of Number of Nodes")
    ax1.set_xlabel("Number of Nodes")
    ax1.set_ylabel("Frequency")
    ax1.set_xticks(range(min(training_node_counts), max(training_node_counts) + 1))

    ax2.hist(test_node_counts, bins=10, edgecolor="black")
    ax2.set_title("Test Data: Histogram of Number of Nodes")
    ax2.set_xlabel("Number of Nodes")
    ax2.set_ylabel("Frequency")
    ax2.set_xticks(range(min(test_node_counts), max(test_node_counts) + 1))

    # plt.tight_layout()
    plt.show()
