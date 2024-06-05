import networkx as nx

task_graph = nx.DiGraph()
task_graph.add_node(0, require=0, generate=5, processing_time=4)
task_graph.add_node(1, require=3, generate=0, processing_time=4)
task_graph.add_node(2, require=2, generate=0, processing_time=4)
# task_graph.add_node(3, require=4, generate=0, processing_time=5)

task_graph.add_edge(0, 1)
task_graph.add_edge(1, 2)
# task_graph.add_edge(0, 3)

# mapping = {task: processing_element}
mapping_dict = {0: 16, 1: 22, 2: 19, 3: 24}
pe_pos_dict = {16: (0, 0), 22: (2, 1), 19: (3, 0), 24: (0, 2)}
pos = nx.spring_layout(task_graph, seed=42)

generate_labels = nx.get_node_attributes(task_graph, "generate")
require_labels = nx.get_node_attributes(task_graph, "require")

processing_time_labels = nx.get_node_attributes(task_graph, "processing_time")
labels = {
    node: f"{node}\ng:{generate_labels.get(node, '')} r:{require_labels.get(node, '')}, t:{processing_time_labels.get(node, '')}"
    for node in task_graph.nodes()
}

nx.draw(task_graph, pos, with_labels=False, node_color="lightblue", node_size=500)
nx.draw_networkx_labels(task_graph, pos, labels=labels)

# import matplotlib.pyplot as plt
# plt.show()

from processing_element import ProcessingElement
from router import Router

list_of_pe = []
list_of_router = []
for node in task_graph.nodes():
    generate = generate_labels.get(node, 0)
    require = require_labels.get(node, 0)
    processing_time = processing_time_labels.get(node, 0)
    connected_nodes = list(task_graph.successors(node))
    connected_pe_list = []

    for connected_node in connected_nodes:
        connected_pes_dict = {}
        connected_pes_dict["pe"] = mapping_dict[connected_node]
        connected_pes_dict["pos"] = pe_pos_dict[mapping_dict[connected_node]]
        connected_pes_dict["require"] = require_labels.get(connected_node, 0)
        connected_pe_list.append(connected_pes_dict)

    pe_id = mapping_dict[node]
    print(f"\nPE ID: {pe_id}")
    pe_pos = pe_pos_dict[pe_id]
    print(
        f"Node {node} requires {require} packets and generates {generate} packets"
        f" with processing time {processing_time} cycles"
    )
    print(f"Connected Nodes: {connected_nodes}")
    print(f"Connected PEs: {connected_pe_list}")

    pe = ProcessingElement(
        pe_pos, require, generate, processing_time, connected_pe_list
    )
    router = Router(pe_pos, 4)

    list_of_pe.append(pe)
    list_of_router.append(router)

print(f"List of Processing Elements: {list_of_pe}")

max_cycles = 50
print(f"\nSimulating for {max_cycles} cycles")

for cycle in range(max_cycles):
    print(f"\n----------------Cycle: {cycle}----------------")
    for pe, router in zip(list_of_pe, list_of_router):
        print(f"\n{pe}, {router}")
        print(f"PRE:\t {pe.pe_status}" f"\n|\n-(Process)")
        # pe.recieve_packets()
        packet = pe.process()
        if packet is not None:
            print(f"Transmitted Packet: {packet}")
            router.add_packet_to_network_interface(packet)
        print(
            f"-(EndOf)\n|"
            f"\nPOST:\t {pe.pe_status}, "
            f"\tProcessing Cycle:\t {pe.current_processing_cycle}/{pe.processing_cycle}, "
            f"\t\tGenerated:\t {pe.generated_count}/{pe.generate} packets"
        )
        print(
            f"Tx:\t {pe.transmitting_status}, "
            f"\tFlites Transmitted:\t {pe.flits_transmitted_count}/{pe.packet_size_in_flits}, "
            f"\tPacket Transmitted:\t {pe.packet_transmitted_count}/{pe.generate}"
        )
