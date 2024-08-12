from data.utils import load_graph_from_json

import os
from natsort import natsorted
import matplotlib.pyplot as plt
import networkx as nx


def calculate_average_degree(graph):
    return sum(dict(graph.degree()).values()) / len(graph.nodes)


def calculate_density(graph):
    return nx.density(graph)


if __name__ == "__main__":

    training_data_path = "data/training_data/input"
    test_data_path = "data/training_data/test/input"

    training_input_files = natsorted(os.listdir(training_data_path))
    test_files = natsorted(os.listdir(test_data_path))

    training_node_counts = []
    training_avg_degrees = []
    training_densities = []
    for file in training_input_files:
        graph = load_graph_from_json(os.path.join(training_data_path, file))
        training_node_counts.append(len(graph.nodes))
        training_avg_degrees.append(calculate_average_degree(graph))
        training_densities.append(calculate_density(graph))

    test_node_counts = []
    test_avg_degrees = []
    test_densities = []
    for file in test_files:
        graph = load_graph_from_json(os.path.join(test_data_path, file))
        test_node_counts.append(len(graph.nodes))
        test_avg_degrees.append(calculate_average_degree(graph))
        test_densities.append(calculate_density(graph))

    overall_training_avg_degree = sum(training_avg_degrees) / len(training_avg_degrees)
    overall_training_density = sum(training_densities) / len(training_densities)
    overall_test_avg_degree = sum(test_avg_degrees) / len(test_avg_degrees)
    overall_test_density = sum(test_densities) / len(test_densities)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.hist(training_node_counts, bins=10, edgecolor="black")
    ax1.set_title("Training Data: Histogram of Number of Nodes")
    ax1.set_xlabel("Number of Nodes")
    ax1.set_ylabel("Frequency")
    ax1.set_xticks(range(min(training_node_counts), max(training_node_counts) + 1))
    ax1.text(
        0.95, 0.95, f'Avg Degree: {overall_training_avg_degree:.2f}\nDensity: {overall_training_density:.2f}', 
        transform=ax1.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right'
    )

    ax2.hist(test_node_counts, bins=10, edgecolor="black")
    ax2.set_title("Test Data: Histogram of Number of Nodes")
    ax2.set_xlabel("Number of Nodes")
    ax2.set_ylabel("Frequency")
    ax2.set_xticks(range(min(test_node_counts), max(test_node_counts) + 1))
    ax2.text(
        0.95, 0.95, f'Avg Degree: {overall_test_avg_degree:.2f}\nDensity: {overall_test_density:.2f}', 
        transform=ax2.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right'
    )

    # plt.tight_layout()
    plt.show()