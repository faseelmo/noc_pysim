
import os
import json

import networkx as nx

from natsort import natsorted
from dataclasses    import dataclass 

from data.utils import (
    load_graph_from_json, 
    visualize_graph, 
    get_compute_list_from_json)


@dataclass
class DependOn:
    id                 : int
    end_cycle          : int

@dataclass
class Node:
    id                  : int
    type                : str

    processing_time     : int = None
    generate            : int = None

    minimum_start_cycle : int = None
    actual_start_cylce  : int = None
    end_cycle           : int = None

    depend_list         : list[DependOn] = None
    is_assigned         : bool = False

PACKET_SIZE = 4 # flits
    
def update_depend_on_end_cycle_for_all_nodes(node_list: list[Node], executed_node:Node) -> None:
    """ 
    Check if the executed node is a dependent node for any other node
    if yes, then update DependOn.end_cycle """
    for node in node_list:

        if node.is_assigned:
            continue

        for depend in node.depend_list:
            if depend.id == executed_node.id:
                depend.end_cycle = executed_node.end_cycle

def assign_minimum_start_cycle_as_depend_cycle(node_list: list[Node]) -> None:
    """ 
    For nodes that have full dependency list [DependOn.end_cycle != None] 
    Assign the minimum start_cycle to the DependOn.end_cycle """
    for node in node_list:

        if node.is_assigned:
            continue

        all_depend_have_end_cycle = False

        for depend in node.depend_list:

            if depend.end_cycle is None:
                all_depend_have_end_cycle = False
                break

            all_depend_have_end_cycle = True

        if all_depend_have_end_cycle:
            # Assign the maximum end_cycle of the dependecy list to the minimum_start_cycle
            node.minimum_start_cycle = max([depend.end_cycle for depend in node.depend_list])

def get_list_of_successor_nodes_as_node_obj(successor_nodes: list, graph: nx.DiGraph, node_id: int) -> list:

    # Assign minimum_wait_time to all the successor nodes
    list_of_successor_nodes = []
    
    for successor_node in successor_nodes:
        require             = graph.edges[node_id, successor_node]['weight']
        minimum_start_cycle = PACKET_SIZE * require

        # Check if the successor node has any other dependency
        dependent_list = []
        has_other_dependency = False
        predecessor_nodes    = list(graph.predecessors(successor_node))

        for predecessor_node in predecessor_nodes:

            if graph.nodes[predecessor_node]['type'] != "dependency":

                depend = DependOn(id = predecessor_node, end_cycle = None)
                has_other_dependency = True
                dependent_list.append(depend)

        if has_other_dependency:
            minimum_start_cycle = None

        # Creating a node object
        node = Node(id = successor_node,
                    type = graph.nodes[successor_node]['type'],
                    processing_time=graph.nodes[successor_node]['processing_time'],
                    generate=graph.nodes[successor_node]['generate'],
                    minimum_start_cycle = minimum_start_cycle,
                    actual_start_cylce = None, 
                    end_cycle = None,
                    depend_list=dependent_list,
                    )

        list_of_successor_nodes.append(node)

    return list_of_successor_nodes

def init_first_dependency_node( graph: nx.DiGraph ) -> tuple[list, int]:
    """
    Find the dependecy node, 
    Iterate through the successor nodes and assign minimum_start_cycle (if applicable)
    Find the node that will start the earliest
    """

    # Assign minimum_wait_time to all applicable dependent nodes
    # and finding the node that will start the earliest
    for idx, node in graph.nodes(data=True):

        if node["type"] == "dependency":

            successor_nodes         = list(graph.successors(idx))

            list_of_successor_nodes = get_list_of_successor_nodes_as_node_obj(successor_nodes, graph, idx)

            # Finding the node with least minimum_start_cycle
            # to assign the actual_start_cycle
            start_cycle_node_idx_value = (None, None) # (index, start_cycle)
            init_assign_flag = False
            for idx, node in enumerate(list_of_successor_nodes):

                if len(node.depend_list) != 0:
                    # You can also do node.minimum_start_cycle is None
                    continue

                if not init_assign_flag:
                    init_assign_flag = True
                    start_cycle_node_idx_value = (idx, node.minimum_start_cycle)

                if node.minimum_start_cycle < start_cycle_node_idx_value[1]:
                    start_cycle_node_idx_value = (idx, node.minimum_start_cycle)

            # Assigning the actual_start_cycle
            list_of_successor_nodes[start_cycle_node_idx_value[0]].actual_start_cylce = start_cycle_node_idx_value[1]

            assert list_of_successor_nodes[start_cycle_node_idx_value[0]].actual_start_cylce == list_of_successor_nodes[start_cycle_node_idx_value[0]].minimum_start_cycle

    return list_of_successor_nodes, start_cycle_node_idx_value[0]
    
def update_end_cylce(executed_node: Node) -> Node: 
    """Computing the end cycle with processing time and generate and actual_start_cycle"""
    executed_node.end_cycle = executed_node.actual_start_cylce + ( executed_node.processing_time * executed_node.generate ) 
    executed_node.is_assigned = True

    return executed_node

def node_list_has_all_nodes(node_list: list[Node], num_nodes: int) -> bool:
    if len(node_list) != num_nodes:
        return False
    return True

def check_all_nodes_assigned(node_list: list[Node]) -> bool:
    for node in node_list:
        if not node.is_assigned:
            return False
    return True

def check_if_node_has_successor(graph: nx.DiGraph, node_id: int) -> tuple[bool, list]:

    successor_nodes = list(graph.successors(node_id))
    if len(successor_nodes) == 0:
        return False, []

    return True, successor_nodes

def is_node_in_node_list(node_list: list[Node], node_id: int) -> bool:

    for node in node_list:
        if node.id == node_id:
            return True

    return False

def get_node_with_least_minimum_start_cycle(node_list: list[Node]) -> Node:

    print(f"Nodes ready to execute {node_list}")

    minimum_index = None
    first_flag = True

    for idx, node in enumerate(node_list):

        if node.is_assigned:
            continue

        if node.minimum_start_cycle is None:
            continue

        if first_flag:
            minimum_index = idx

        if node.minimum_start_cycle < node_list[minimum_index].minimum_start_cycle:
            minimum_index = idx 

    return node_list[minimum_index]

def update_depend_on_for_new_node_list(node_list: list[Node], new_node_list: list[Node]) -> None:    

    for new_node in new_node_list: 

        for depend in new_node.depend_list:
            depend_id = depend.id

            for node in node_list:

                if not node.is_assigned:
                    continue
                if node.id == depend_id:
                    depend.end_cycle = node.end_cycle

def get_list_nodes_ready_to_execute(node_list: list[Node]) -> list[Node]: 

    list_of_nodes_ready_to_execute = []
    for node in node_list:

        if node.is_assigned:
            continue

        all_depend_have_end_cycle = False

        if len(node.depend_list) != 0:
            for depend in node.depend_list:

                if depend.end_cycle is None:
                    all_depend_have_end_cycle = False
                    break

                all_depend_have_end_cycle = True

        else: 
            all_depend_have_end_cycle = True

        if all_depend_have_end_cycle:
            list_of_nodes_ready_to_execute.append(node)

    return list_of_nodes_ready_to_execute
    

def assign_cycle_for_executing_node(node_list: list[Node], node_id: int) -> None:

    # Finding max end cycle of all the assigned nodes
    list_of_start_cycles = []
    for node in node_list:

        if not node.is_assigned:
            continue

        list_of_start_cycles.append(node.end_cycle)

    max_end_cycle_of_assigned_nodes = max(list_of_start_cycles)


    # Finding the max of dependency end cycle
    list_of_depend_end_cycles = []
    for node in node_list:
        if node.id == node_id:
            for depend in node.depend_list:
                for node in node_list:
                    if node.id == depend.id:
                        list_of_depend_end_cycles.append(node.end_cycle)
    

    if len(list_of_depend_end_cycles) == 0:
        start_cycle = max_end_cycle_of_assigned_nodes
    else: 
        max_end_cycle_of_depend_nodes = max(list_of_depend_end_cycles)
        start_cycle = max(max_end_cycle_of_assigned_nodes, max_end_cycle_of_depend_nodes)

    for node in node_list:
        if node.id == node_id:
            node.actual_start_cylce = start_cycle + 1


def get_minimum_start_cycle_from_assigned_nodes(node_list: list[Node]) -> int:

    list_of_start_cycles = []

    for node in node_list:

        if not node.is_assigned:
            continue

        list_of_start_cycles.append(node.end_cycle)

    return max(list_of_start_cycles)


def main(graph):
    
    num_nodes = graph.number_of_nodes()    
    
    node_list, start_node_index = init_first_dependency_node(graph)

    start_node = node_list[start_node_index]

    executed_node = update_end_cylce(start_node) 

    update_depend_on_end_cycle_for_all_nodes(node_list, executed_node)

    assign_minimum_start_cycle_as_depend_cycle(node_list)

    # Checking if the nodes in node_list has all its dependendents in the node_list
    new_node_list = []
    for node in node_list:

        has_successor, successor_nodes = check_if_node_has_successor(graph, node.id)
        if has_successor:

            for successor_node in successor_nodes:

                if not is_node_in_node_list(node_list, successor_node):
                    
                    if not is_node_in_node_list(new_node_list, successor_node):
                        list_of_new_nodes = get_list_of_successor_nodes_as_node_obj(successor_nodes, graph, node.id)
                        new_node_list.extend(list_of_new_nodes)

    update_depend_on_for_new_node_list(node_list, new_node_list)
    node_list.extend(new_node_list)


    while not check_all_nodes_assigned(node_list):
        assign_minimum_start_cycle_as_depend_cycle(node_list)
        nodes_ready_to_execute  = get_list_nodes_ready_to_execute(node_list)

        start_node                      = get_node_with_least_minimum_start_cycle(nodes_ready_to_execute)
        possible_minimum_start_cycle    = get_minimum_start_cycle_from_assigned_nodes(node_list)

        if possible_minimum_start_cycle > start_node.minimum_start_cycle:
            start_node.actual_start_cylce = possible_minimum_start_cycle

        else: 
            start_node.actual_start_cylce = start_node.minimum_start_cycle

        update_end_cylce(start_node)    
        update_depend_on_end_cycle_for_all_nodes(node_list, start_node)


    print(f"\nNodes in node_list")
    for node in node_list:
        print(node)


PACKET_SIZE = 4 # flits

if __name__ == "__main__":

    input_training_data_path    = "data/training_data/input"
    packet_list_training_path   = "data/training_data/packet_list"
    target_training_data_path   = "data/training_data/target"

    input_test_data_path        = "data/training_data/test/input"
    packet_list_test_path       = "data/training_data/test/packet_list"
    target_test_data_path       = "data/training_data/test/target"

    input_analytical_latency_path   = "data/analytical_test_data/input"
    input_analytical_files          = natsorted(os.listdir(input_analytical_latency_path))

    input_training_data_files   = natsorted(os.listdir(input_training_data_path))
    packet_list_training_files  = natsorted(os.listdir(packet_list_training_path))
    target_training_input_files = natsorted(os.listdir(target_training_data_path))

    input_test_data_files       = natsorted(os.listdir(input_test_data_path))
    packet_list_test_files      = natsorted(os.listdir(packet_list_test_path))
    target_list_test_files      = natsorted(os.listdir(target_test_data_path))

    for input_idx in input_analytical_files:
        
        graph = load_graph_from_json(os.path.join(input_analytical_latency_path, input_idx))

        print(f"\n--------Graph: {input_idx}--------")

        dependency_node_count = 0

        for idx, node in graph.nodes(data=True):
            if node["type"] == "dependency":
                dependency_node_count += 1

        # Skip Graphs with more than 1 dependency node
        if dependency_node_count > 1:
            continue

        
        main(graph)

        visualize_graph(graph)






        
    # for input_idx, target_idx, packet_list_idx in zip(input_training_data_files, target_training_input_files, packet_list_training_files):
        # compute_list    = get_compute_list_from_json(os.path.join(target_training_data_path, target_idx)) 
        # packet_list     = json.load(open(os.path.join(packet_list_training_path, packet_list_idx)))
        # visualize_graph(graph, packet_list=packet_list, compute_list=compute_list)

