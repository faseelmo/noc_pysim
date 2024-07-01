import networkx as nx

from src.utils import graph_to_task_list
from src.utils import get_random_packet_list
from src.utils import simulate

from data.utils import (
    # save_graph_to_json, 
    load_graph_from_json,
    visualize_graph
)

graph = load_graph_from_json("data/test_task_graph.json")
# visualize_graph(graph)

computing_list = graph_to_task_list(graph)
for task in computing_list:
    print(f"\n")
    print(task)

packet_list = get_random_packet_list(graph)
print(f"\nPackets: ")
for packet in packet_list:
    print(packet)

latency = simulate(computing_list, packet_list)
print(f"\nLatency: {latency}")

# for packet in packet_list:
#     print(packet)
