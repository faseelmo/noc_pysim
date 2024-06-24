from src.processing_element import ProcessingElement, TaskInfo, RequireInfo
from src.packet import PacketStatus, Packet
from src.utils import graph_to_task_list
import networkx as nx
import copy

MAX_CYCLES = 100


def pe_status_string(compute_is_busy: bool):
    if compute_is_busy:
        return "Computing"
    else:
        return "Free"


def check_if_pe_is_done(packet_list, pe):
    if len(packet_list) == 0 and not pe.compute_is_busy:
        return True


def create_packet_copies(num_copies, packet_source):
    packet = Packet(source_xy=(0, 0), dest_xy=(1, 1), source_task_id=packet_source)
    return [copy.deepcopy(packet) for _ in range(num_copies)]


def simulate(computing_list: list, packet_list: list):

    pe = ProcessingElement((0, 0), computing_list)
    current_packet = packet_list.pop(0)

    for cycle in range(MAX_CYCLES):
        print(f"\n> {cycle}")
        pe.process(current_packet)
        if not current_packet is None and current_packet.status is PacketStatus.IDLE:
            if len(packet_list) > 0:
                current_packet = packet_list.pop(0)
            else:
                current_packet = None
        print(f"POST: {pe}\t{pe_status_string(pe.compute_is_busy)}")
        if pe.check_task_requirements_met():
            break

    return cycle


def connect_dependecy_node(graph, task_id, dependency_id, require):
    """
    Connects a dependency node to a task node in the graph.
    Require arg is the weight of the edge between the task and dependency node
    """
    graph.add_node(dependency_id, type="dependency")
    graph.add_edge(dependency_id, task_id, weight=require)


def test_task_sequential_1():
    """Test 1"""

    graph = nx.DiGraph()
    graph.add_node(0, type="task", generate=2, processing_time=5)
    connect_dependecy_node(graph, dependency_id=2, task_id=0, require=3)
    connect_dependecy_node(graph, dependency_id=3, task_id=0, require=2)

    graph.add_node(1, type="task", generate=3, processing_time=4)
    graph.add_edge(0, 1, weight=2)

    computing_list = graph_to_task_list(graph)

    packet_2_copies = create_packet_copies(packet_source=2, num_copies=3)
    packet_3_copies = create_packet_copies(packet_source=3, num_copies=2)

    packet_list = [
        packet_2_copies.pop(0),
        packet_2_copies.pop(0),
        packet_3_copies.pop(0),
        packet_3_copies.pop(0),
        packet_2_copies.pop(0),
    ]

    latency = simulate(computing_list, packet_list)
    assert latency == 40


def test_task_sequential_2():
    """Test 2"""

    graph = nx.DiGraph()
    graph.add_node(0, type="task", generate=2, processing_time=5)
    connect_dependecy_node(graph, dependency_id=2, task_id=0, require=3)
    connect_dependecy_node(graph, dependency_id=3, task_id=0, require=2)

    graph.add_node(1, type="task", generate=3, processing_time=4)
    connect_dependecy_node(graph, dependency_id=4, task_id=1, require=5)

    computing_list = graph_to_task_list(graph)

    packet_2_copies = create_packet_copies(packet_source=2, num_copies=3)
    packet_3_copies = create_packet_copies(packet_source=3, num_copies=2)
    packet_4_copies = create_packet_copies(packet_source=4, num_copies=5)

    packet_list = [
        packet_2_copies.pop(0),
        packet_2_copies.pop(0),
        packet_3_copies.pop(0),
        packet_3_copies.pop(0),
        packet_2_copies.pop(0),
        *packet_4_copies,
    ]

    latency = simulate(computing_list, packet_list)
    assert latency == 50


def test_task_parallel_3():
    """Test 3"""

    graph = nx.DiGraph()    
    graph.add_node(0, type="task", generate=2, processing_time=7)
    connect_dependecy_node(graph, dependency_id=2, task_id=0, require=1)
    connect_dependecy_node(graph, dependency_id=3, task_id=0, require=1)

    graph.add_node(1, type="task", generate=3, processing_time=4)
    connect_dependecy_node(graph, dependency_id=4, task_id=1, require=2)

    computing_list = graph_to_task_list(graph)


    packet_2_copies = create_packet_copies(packet_source=2, num_copies=1)
    packet_3_copies = create_packet_copies(packet_source=3, num_copies=1)
    packet_4_copies = create_packet_copies(packet_source=4, num_copies=2)

    packet_list = [*packet_2_copies, *packet_3_copies, *packet_4_copies]

    latency = simulate(computing_list, packet_list)
    assert latency == 32


