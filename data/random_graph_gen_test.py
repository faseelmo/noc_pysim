import networkx as nx
import matplotlib.pyplot as plt

"""
useable graph from nx
https://networkx.org/documentation/stable/reference/generators.html#module-networkx.generators.directed
"""

num_nodes = 2  # Example number of nodes

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 18))
plt.subplots_adjust(hspace=0.5)

# GN Graph
graph = nx.gn_graph(num_nodes, seed=42)
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_size=300, node_color="skyblue", arrows=True, ax=axes[0, 0])
axes[0, 0].set_title("GN Graph")

# GNR Graph
graph = nx.gnr_graph(num_nodes, 1, seed=42)
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_size=300, node_color="skyblue", arrows=True, ax=axes[0, 1])
axes[0, 1].set_title("GNR Graph")

# GNC Graph
graph = nx.gnc_graph(num_nodes, seed=42)
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_size=300, node_color="skyblue", arrows=True, ax=axes[1, 0])
axes[1, 0].set_title("GNC Graph")

# Random k out Graph
graph = nx.random_k_out_graph(num_nodes, 2, 0.5, seed=42)
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_size=300, node_color="skyblue", arrows=True, ax=axes[1, 1])
axes[1, 1].set_title("Random K Out Graph")

# Scale Free Graph
graph = nx.scale_free_graph(num_nodes, seed=42)
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_size=300, node_color="skyblue", arrows=True, ax=axes[2, 0])
axes[2, 0].set_title("Scale Free Graph")

# Adjust layout for the last subplot which is unused
fig.delaxes(axes[2][1])

# plt.savefig("random_graphs_2.png")
plt.show()