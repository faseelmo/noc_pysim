import json
import random
import networkx as nx

from src.simulator import Map
from src.packet import PacketStatus, Packet
from src.processing_element import TaskInfo

def graph_to_task_list(graph: nx.DiGraph) -> list:
    """
    Converts a graph to a list of tasks
    Needs graph with nodes having the following attributes:
    1. node_type = "task" or "dependency"
        if node_type = "task":
            generate = int
            processing_time = int

    2. Between task and dependency nodes,
        there should be an edge with weight = int,
        representing the number of packets required

    """
    from src.processing_element import TaskInfo
    from src.processing_element import RequireInfo

    computing_list = []
    # Creating computing list
    for node in graph.nodes:

        if graph.nodes[node]["type"] in ("task", "task_depend"):

            task = TaskInfo(
                task_id=node,
                processing_cycles=graph.nodes[node]["processing_time"],
                expected_generated_packets=graph.nodes[node]["generate"],
                require_list=[],
            )
            computing_list.append(task)

    # Updating require list for each task
    for node in graph.nodes:

        if graph.nodes[node]["type"] in ("task", "task_depend"):
            predecessors = list(graph.predecessors(node))

            for predecessor in predecessors:
                # print(f"Node {node} has predecessor {predecessor}")
                for task in computing_list:

                    if task.task_id == node:
                        require = RequireInfo(
                            require_type_id=predecessor,
                            required_packets=graph[predecessor][node]["weight"],
                        )
                        task.require_list.append(require)

    return computing_list

def simulate_application_on_pe( computing_list  : list[TaskInfo], 
                                packet_list     : list[Packet], 
                                debug_mode      : bool = False, 
                                sjf_scheduling  : bool = False ) -> int:
    """
    Simulation of the processing element

        - Each processing element is intialized with the `computing_list`
          and the (x,y) position of the processing element in the mesh

        - A for loop is run for `max_cycles` simulating the cycles

        - In each cycle, the processing element processes a packet by
          passing in the packet to `process()`. The packets come from the 
          `packet_list`. 

        - Packets initially are in the `IDLE` state. During transmission, 
          Packet status is changed to `TRANSMITTING` and once the packet
          is fully transmitted, the status is changed back to `IDLE`. 

        - `computing_list` is updated with start_cycle and end_cycle. 
    """

    from src.processing_element import ProcessingElement

    pe = ProcessingElement((0, 0), computing_list, debug_mode=debug_mode, shortest_job_first=sjf_scheduling)
    current_packet = packet_list.pop(0)
    cycle = 0

    while True: 

        if debug_mode:
            print(f"\n> {cycle}")

        is_done_processing = pe.process(current_packet)

        current_packet = update_current_packet(current_packet, packet_list)

        if is_done_processing:
            break

        cycle += 1

    return cycle


def update_current_packet(current_packet: Packet, packet_list: list[Packet]) -> Packet:
    """
    Updates the current packet if it is idle and there are packets in the packet list.
    """
    if current_packet is not None and current_packet.get_status() is PacketStatus.IDLE:
        if len(packet_list) > 0:
            current_packet = packet_list.pop(0)
        else:
            current_packet = None
    return current_packet


def get_ordered_packet_list(graph: nx.DiGraph) -> list:
    """
    Packets ordered based on wait_time of task_depend
    lowest wait_time first 
    """

    from dataclasses import dataclass
    from src.packet import Packet

    @dataclass
    class TaskDepend: 
        minimum_wait_time   : int
        types_of_packets    : list[int]

    @dataclass
    class PacketCount: 
        id: int 
        max_count : int


    task_depend_nodes = []
    packet_count_list = []

    for node_id, node in graph.nodes(data=True):

        if node["wait_time"] != 0:
            # Fills the task_depend_nodes list

            wait_time       = node["wait_time"]
            packet_types    = []
            predecessors    = list(graph.predecessors(node_id))
            
            for predecessor_id in predecessors:
                if graph.nodes[predecessor_id]["type"] == "dependency":
                    packet_types.append(predecessor_id)

            task_depend = TaskDepend(
                            minimum_wait_time   = wait_time,
                            types_of_packets    = packet_types )

            task_depend_nodes.append(task_depend)

        if node["type"] == "dependency":
            # Fills the packet_count_list

            packet_id   = node_id
            maxcount    = node["generate"]

            packet = PacketCount(
                id=packet_id,
                max_count=maxcount
            )

            packet_count_list.append(packet)

    # Sort the task_depend_nodes by minimum_wait_time. Asceding order. 
    ordered_task_depend_nodes = sorted(task_depend_nodes, key=lambda x: x.minimum_wait_time)

    # Also sorting the packet list by the order of the task_depend_nodes
    # and removing duplicates
    ordered_packet_list = []
    for task_depend in ordered_task_depend_nodes:

        for packet in packet_count_list:
            if packet.id in task_depend.types_of_packets:
                
                if packet.id not in [packet.id for packet in ordered_packet_list]:
                    ordered_packet_list.append(packet)

    # Generate packets for each packet in the ordered_packet_list
    # upto the max_count
    packet_list = []
    for ordered_packet in ordered_packet_list:
        for i in range(ordered_packet.max_count):

            packet = Packet(source_xy=(0, 0), dest_id=ordered_packet.id, source_task_id=ordered_packet.id)
            packet_list.append(packet)  

    return packet_list


def get_random_packet_list(graph: nx.DiGraph, shuffle=False) -> list:
    # > Look into here

    from src.processing_element import RequireInfo
    from src.packet import Packet

    packet_list = []
    required_packet_type = []  # (type,count)

    for node in graph.nodes:
        predecessors = list(graph.predecessors(node))
        successors = list(graph.successors(node))

        if len(predecessors) == 0:
            # for dependency nodes
            successor_require = []
            for successor in successors:
                edge_data = graph.get_edge_data(node, successor)
                weight = edge_data["weight"]
                successor_require.append(weight)

            # We take the max from all the requirements
            # since we have cache now to copy
            num_packets = max(successor_require)

            require = RequireInfo(
                require_type_id=node,
                required_packets=num_packets,
            )
            required_packet_type.append(require)

    for require in required_packet_type:
        for i in range(require.required_packets):
            packet = Packet(
                source_xy=(0, 0), dest_id=1, source_task_id=require.require_type_id
            )
            packet_list.append(packet)

    if shuffle:
        random.shuffle(packet_list)

    return packet_list

def get_graph_report(graph: nx.DiGraph, mapping_list: list[Map], show: bool = False) -> nx.DiGraph:
    """
    Updates the node of the application graph with the processing start_cycle and end_cycle.
    """
    for node_id, node in graph.nodes(data=True):

        for map in mapping_list:
            task = map.task
            if task.task_id == node_id:
                start_cycle = task.start_cycle
                end_cycle   = task.end_cycle    
                node["start_cycle"] = start_cycle
                node["end_cycle"]   = end_cycle
                node["assigned_pe"] = map.assigned_pe

    return graph

def does_nodes_have_types(graph: nx.DiGraph) -> bool:

    for node in graph.nodes:
        if "type" not in graph.nodes[node]:
            return False
    
    return True


def get_mesh_network(mesh_size: int, application_graph: nx.DiGraph, mapping_list: list[Map]) -> nx.DiGraph: 

    import re 
    import networkx as nx

    if not does_nodes_have_types(application_graph):
        application_graph = assign_types_to_nodes(application_graph)

    # Creating the Router network graph
    graph       = nx.grid_2d_graph(mesh_size, mesh_size, create_using=nx.DiGraph)
    rename_map  = {node: f"Router({node[0]},{node[1]})" for node in graph.nodes()}
    graph       = nx.relabel_nodes(graph, rename_map, copy=False)
    nx.set_node_attributes(graph, "router", "type")

    router_nodes = list(graph.nodes())  

    # Adding the processing elements (PEs) to the mesh network graph
    for router_str in router_nodes: 
        x, y = tuple(map(int, re.findall(r'\d+', router_str)))
        pe_node = f"PE({x},{y})"            # Create a label for each PE node
        graph.add_node(pe_node, type="pe")  # Add the PE node
        graph.add_edge(pe_node, router_str) # Connect PE to the router (directed edge)
        graph.add_edge(router_str, pe_node) # Connect the router to the PE (directed edge)

    # Adding the tasks from the application graph to mesh network graph
    # + create the mapping edge between the task and the PE
    for task_id in application_graph.nodes():
        for map_ in mapping_list:
            if map_.task.task_id == task_id:

                graph.add_node(
                    task_id, 
                    type            = "task", 
                    task_type       = application_graph.nodes[task_id]["type"],  
                    generate        = map_.task.expected_generated_packets, 
                    processing_time = map_.task.processing_cycles, 
                    start_cycle     = map_.task.start_cycle, 
                    end_cycle       = map_.task.end_cycle)

                pe_x, pe_y = map_.assigned_pe
                graph.add_edge(task_id, f"PE({pe_x},{pe_y})")
                # graph.add_edge(f"PE({pe_x},{pe_y})", task_id)

    # Adding edges of the application graph in the mesh network graph
    for task_id in application_graph.nodes():
        successors = list(application_graph.successors(task_id))
        for successor in successors:
            weight = application_graph[task_id][successor]["weight"]
            graph.add_edge(task_id, successor, weight=weight)

    return graph


def load_graph_from_json(filename: str) -> nx.DiGraph:
    with open(filename, "r") as file:
        data = json.load(file)
    return nx.node_link_graph(data)

def save_graph_to_json(graph: nx.DiGraph, filename: str):
    data = nx.node_link_data(graph)
    with open(filename, "w") as file:
        json.dump(data, file)


def visualize_noc_application(graph: nx.DiGraph, prediction: list = None):
    import matplotlib.pyplot  as plt
    import numpy as np
    import re

    has_prediction = False
    if prediction: 
        assert isinstance(prediction, list), "Prediction should be a list"
        has_prediction = True

    task_color_map  = { "dependency": "mediumpurple", "task": "gold", "exit": "tomato"}
    noc_color_map   = { "router": "royalblue", "pe": "mediumseagreen" }
    node_colors     = []

    for node in graph.nodes:
        if graph.nodes[node].get("type") == "task":
            node_type   = graph.nodes[node]["task_type"]
            color       = task_color_map[node_type]
            node_colors.append(color)

        else: 
            node_type   = graph.nodes[node]["type"]
            color       = noc_color_map[node_type]
            node_colors.append(color)   

    router_tilt                 = 0.4
    pe_offset                   = 0.2
    normalization_factor        = 3 
    application_graph_offset    = 4.0

    pos         = {}
    task_nodes  = [node for node, data in graph.nodes(data=True) if data.get('type') == 'task']
    task_pos    = nx.spring_layout(graph.subgraph(task_nodes), seed=0)
    pos.update(task_pos)

    # Finding normalized positions for the task nodes
    x_values = np.array([pos[0] for pos in pos.values()])
    y_values = np.array([pos[1] for pos in pos.values()])

    x_min, x_max = x_values.min(), x_values.max()
    y_min, y_max = y_values.min(), y_values.max()

    x_values = normalization_factor * (x_values - x_min) / (x_max - x_min)
    y_values = normalization_factor * (y_values - y_min) / (y_max - y_min)

    y_offset = application_graph_offset  
    y_values += y_offset

    # Updating pos with normalized task positions
    for i, node in enumerate(pos.keys()):
        pos[node] = (x_values[i], y_values[i])

    # Updating the positions of the router and PE nodes
    for node_str in graph.nodes():
        if graph.nodes[node_str].get('type') == 'router':
            x, y                = tuple( map( int, re.findall(r'\d+', node_str) ) ) 
            pos[node_str]       = ( x + router_tilt * y, y )
            pos[f"PE({x},{y})"] = ( x + (router_tilt * y) + pe_offset, y + pe_offset )

    # Custom labels for the nodes
    custom_labels = {}
    for id, node in graph.nodes(data=True): 
        label       = [f"id: {id}"]
        node_type   = node.get('type')

        if node_type == "task":
            label.append(f"True: {node.get('start_cycle', 'N/A')} -> {node.get('end_cycle', 'N/A')}")
            if has_prediction:
                start = int(prediction[id][0])
                end  = int(prediction[id][1])
                label.append(f"Pred: {start} -> {end}")

        custom_labels[id] = "\n".join(label)

    plt.figure(figsize=(10, 10))
    nx.draw(graph, 
            pos, 
            with_labels = True, 
            node_color  = node_colors, 
            node_size   = 500, 
            labels      = custom_labels, 
            font_size   = 10, 
            font_weight = 'bold', 
            edge_color  = 'gray')

    pe_nodes = [n for n in graph.nodes() if isinstance(n, str) and n.startswith("PE")]

    nx.draw_networkx_nodes(graph, pos, nodelist=pe_nodes, node_color='lightgreen', node_size=300)
    edge_labels = nx.get_edge_attributes(graph, 'weight')

    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')
    plt.show()

def visualize_application(
        graph: nx.DiGraph, 
        latency_value=None, 
        packet_list=None, 
        compute_list=None, 
        pred_compute_list=None):
    """
    args: 
        compute_list: list or dict. Used to display the start and end cycle (truth) of each task

    """
    import matplotlib.pyplot as plt

    if not does_nodes_have_types(graph):
        graph = assign_types_to_nodes(graph)

    node_color_map = {  "dependency": "mediumpurple", "task": "gold", "exit": "tomato", "scheduler": "mediumseagreen"}

    node_colors = [
        node_color_map.get(graph.nodes[node].get("type", "task"), "lightgreen")
        for node in graph.nodes
    ]

    seed = 0
    pos = nx.spring_layout(graph, seed=seed)
    # color_light_gray = 
    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_size=900,
        node_color=node_colors,
        arrows=True,
        edge_color="lightgray",
    )

    if isinstance(compute_list, dict): 
        node_cycle_dict = compute_list
    else: 
        node_cycle_dict = compute_list_to_node_dict(compute_list) if compute_list else {}

    custom_labels = {}
    for node in graph.nodes:
        label_parts = [f"id: {node}"]
        if "processing_time" in graph.nodes[node]:
            label_parts.append(f"P: {graph.nodes[node]['processing_time']}")
        if "generate" in graph.nodes[node]:
            label_parts.append(f"G: {graph.nodes[node]['generate']}")
        # if "wait_time" in graph.nodes[node]:
        #     wait_time = graph.nodes[node]["wait_time"]
        #     if wait_time != 0:
        #         label_parts.append(f"W: {wait_time}")
            
        if "start_cycle" in graph.nodes[node] and "end_cycle" in graph.nodes[node]: 
            start_time = graph.nodes[node]["start_cycle"]
            end_time   = graph.nodes[node]["end_cycle"]
            label_parts.append(f"{start_time} to {end_time}")

        if node in node_cycle_dict:
            
            truth_start_cycle   = node_cycle_dict[node]['start_cycle']
            truth_end_cycle     = node_cycle_dict[node]['end_cycle']

            label_parts.append(f"T: ({truth_start_cycle}-{truth_end_cycle}) "
                               f"= {truth_end_cycle - truth_start_cycle}")

            if pred_compute_list:
                pred_start_cycle    = pred_compute_list[node]['start_cycle']
                pred_end_cycle      = pred_compute_list[node]['end_cycle']
    
                label_parts.append(f"P: ({pred_start_cycle}-{pred_end_cycle}) "
                                   f"= {pred_end_cycle - pred_start_cycle}")

        custom_labels[node] = "\n".join(label_parts)

    nx.draw_networkx_labels(graph, pos, labels=custom_labels)

    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    if latency_value is not None:
        plt.text(
            0.5,
            0.05,
            f"Latency: {latency_value}",
            ha="center",
            va="bottom",
            transform=plt.gca().transAxes,
        )

    if packet_list is not None:
        plt.text(
            0.5,
            0.01,
            f"Packet list: ←{packet_list}",
            ha="center",
            va="bottom",
            transform=plt.gca().transAxes,
        )

    plt.show()

def get_compute_list_from_json(filename: str) -> dict:
    """
    Converts the node cycle information from the json file to a dictionary
    used in inspect_data.py
    """
    json_dict = json.load(open(filename))

    compute_list = {}
    for key in json_dict:
        if key == "latency":
            continue
        else: 
            compute_list[int(key)] = json_dict[key] 

    return compute_list


def compute_list_to_node_dict(compute_list):
    """
    Converts compute list to a dictionary with task_id as key
    and start and end cycle as value
    """
    node_dict = {}
    for task in compute_list:
        node_dict[task.task_id] = {
            "start_cycle": task.start_cycle,
            "end_cycle": task.end_cycle
        }
    return node_dict

def assign_types_to_nodes(graph: nx.DiGraph):

    for node in graph.nodes:
        successors = list(graph.successors(node))
        predecessors = list(graph.predecessors(node))

        if len(predecessors) == 0:
            graph.nodes[node]["type"] = "dependency"
        elif len(successors) == 0:
            graph.nodes[node]["type"] = "exit"
        else:
            graph.nodes[node]["type"] = "task"

    return graph