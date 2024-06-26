import networkx as nx

from src.utils import graph_to_task_list
from src.utils import get_random_packet_list
from src.utils import simulate

from data.utils import (
    save_graph_to_json, 
    load_graph_from_json,
    visualize_graph
)

graph = load_graph_from_json("data/test_task_graph.json")

packet_list = get_random_packet_list(graph)

for packet in packet_list:
    print(packet)

visualize_graph(graph)