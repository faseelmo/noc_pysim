import copy
import random
import networkx as nx

from src.packet import Packet
from src.utils import graph_to_task_list, get_random_packet_list, simulate_application_on_pe

from src.utils import load_graph_from_json 


MAX_CYCLES = 1000


def pe_status_string(compute_is_busy: bool):
    if compute_is_busy:
        return "Computing"
    else:
        return "Free"


def check_if_pe_is_done(packet_list, pe):
    if len(packet_list) == 0 and not pe.compute_is_busy:
        return True


def create_packet_copies(num_copies, packet_source):
    packet = Packet(source_xy=(0, 0), dest_id=1, source_task_id=packet_source)
    return [copy.deepcopy(packet) for _ in range(num_copies)]


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

    latency = simulate_application_on_pe(computing_list, packet_list)
    assert latency == 42


def test_task_sequential_2():

    graph = nx.DiGraph()
    graph.add_node(0, type="task", generate=2, processing_time=5)
    connect_dependecy_node(graph, dependency_id=2, task_id=0, require=3)
    connect_dependecy_node(graph, dependency_id=3, task_id=0, require=2)

    graph.add_node(1, type="task", generate=3, processing_time=4)
    connect_dependecy_node(graph, dependency_id=4, task_id=1, require=5)

    graph.add_edge(0, 1, weight=2)

    # Test 1 - Task 0 receives the 5 packets first (from 2 and 3)
    test_1_computing_list = graph_to_task_list(graph)
    test_2_computing_list = copy.deepcopy(test_1_computing_list)

    test_1_packet_2_copies = create_packet_copies(packet_source=2, num_copies=3)
    test_1_packet_3_copies = create_packet_copies(packet_source=3, num_copies=2)
    test_1_packet_4_copies = create_packet_copies(packet_source=4, num_copies=5)

    test_1_packet_list = [
        test_1_packet_2_copies.pop(0),
        test_1_packet_2_copies.pop(0),
        test_1_packet_3_copies.pop(0),
        test_1_packet_3_copies.pop(0),
        test_1_packet_2_copies.pop(0),
        *test_1_packet_4_copies,
    ]

    test_1_latency = simulate_application_on_pe(test_1_computing_list, test_1_packet_list)
    assert test_1_latency == 51

    # Test 2 - Task 1 receives the 5 packets first
    test_2_packet_2_copies = create_packet_copies(packet_source=2, num_copies=3)
    test_2_packet_3_copies = create_packet_copies(packet_source=3, num_copies=2)
    test_2_packet_4_copies = create_packet_copies(packet_source=4, num_copies=5)

    test_2_packet_list = [
        *test_2_packet_4_copies,
        test_2_packet_2_copies.pop(0),
        test_2_packet_2_copies.pop(0),
        test_2_packet_3_copies.pop(0),
        test_2_packet_3_copies.pop(0),
        test_2_packet_2_copies.pop(0),
    ]

    test_2_latency = simulate_application_on_pe(test_2_computing_list, test_2_packet_list)
    assert test_2_latency == 62


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

    latency = simulate_application_on_pe(computing_list, packet_list)
    assert latency == 34


def test_task_random_graph_1(): 

    def get_graph_latency(graph_dir):
        graph = load_graph_from_json(graph_dir)
        random.seed(0)
        packet_list = get_random_packet_list(graph)

        packet_list_copy = copy.deepcopy(packet_list)

        computing_list = graph_to_task_list(graph)
        latency = simulate_application_on_pe(computing_list, packet_list, debug_mode=True)

        print(f"PAcket ")
        for packet in packet_list_copy: 
            print(packet)

        return latency

    graph_dir_1 = "tests/test_graphs/task_graph_0.json"
    latency_1 = get_graph_latency(graph_dir_1)
    assert latency_1 == 163

    graph_dir_1 = "tests/test_graphs/task_graph_1.json"
    latency_1 = get_graph_latency(graph_dir_1)
    assert latency_1 == 119




