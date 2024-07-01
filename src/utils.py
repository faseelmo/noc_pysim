import networkx as nx
import random


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
        if graph.nodes[node]["type"] == "task":
            task = TaskInfo(
                task_id=node,
                processing_cycles=graph.nodes[node]["processing_time"],
                expected_generated_packets=graph.nodes[node]["generate"],
                require_list=[],
            )
            computing_list.append(task)

    # Updating require list for each task
    for node in graph.nodes:

        if graph.nodes[node]["type"] == "task":
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


def simulate(computing_list: list, packet_list: list, debug_mode=False):

    from src.processing_element import ProcessingElement
    from src.packet import PacketStatus

    MAX_CYCLES = 1000

    pe = ProcessingElement((0, 0), computing_list, debug_mode=debug_mode)
    current_packet = packet_list.pop(0)

    for cycle in range(MAX_CYCLES):
        if debug_mode: 
            print(f"\n> {cycle}")
        pe.process(current_packet)
        if not current_packet is None and current_packet.status is PacketStatus.IDLE:
            if len(packet_list) > 0:
                current_packet = packet_list.pop(0)
            else:
                current_packet = None
        if pe.check_task_requirements_met():
            break

    return cycle


def get_random_packet_list(graph: nx.DiGraph):

    from src.processing_element import RequireInfo
    from src.packet import Packet

    packet_list = []
    required_packet_type = []  # (type,count)

    for node in graph.nodes:
        predecessors = list(graph.predecessors(node))
        successors = list(graph.successors(node))
        if len(predecessors) == 0:
            num_packets = 0
            for successor in successors:
                edge_data = graph.get_edge_data(node, successor)
                weight = edge_data["weight"]
                num_packets += weight

            require = RequireInfo(
                require_type_id=node,
                required_packets=num_packets,
            )
            required_packet_type.append(require)

    for require in required_packet_type:
        for i in range(require.required_packets):
            packet = Packet(
                source_xy=(0, 0), dest_xy=(1, 1), source_task_id=require.require_type_id
            )
            packet_list.append(packet)

    random.shuffle(packet_list)

    return packet_list
