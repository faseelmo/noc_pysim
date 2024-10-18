import networkx as nx
import random

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


def simulate(
    computing_list: list[TaskInfo], 
    packet_list: list[Packet], 
    debug_mode=False, 
    max_cycles=1000, 
    sjf_scheduling=False
) -> int:
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

    for cycle in range(max_cycles):

        if debug_mode:
            print(f"\n> {cycle}")

        is_done_processing = pe.process(current_packet)

        current_packet = update_current_packet(current_packet, packet_list)

        if is_done_processing:
            break

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

    from dataclasses import dataclass
    from src.packet import Packet

    @dataclass
    class TaskDepend: 
        minimum_wait_time:  int
        types_of_packets:   list[int]

    @dataclass
    class PacketCount: 
        id: int 
        max_count : int


    task_depend_nodes = []
    packet_count_list = []

    for node_id, node in graph.nodes(data=True):

        if node["type"] == "task_depend":
            # Fills the task_depend_nodes list

            wait_time       = node["wait_time"]
            packet_types    = []
            predecessors    = list(graph.predecessors(node_id))
            
            for predecessor_id in predecessors:
                if graph.nodes[predecessor_id]["type"] == "dependency":
                    packet_types.append(predecessor_id)

            task_depend = TaskDepend(
                minimum_wait_time   = wait_time,
                types_of_packets    = packet_types
            )

            task_depend_nodes.append(task_depend)

        if node["type"] == "dependency":
            # Fills the packet_count_list

            packet_id = node_id
            maxcount = node["generate"]

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

            packet = Packet(source_xy=(0, 0), dest_xy=(1, 1), source_task_id=ordered_packet.id)
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

