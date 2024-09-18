
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

    minimum_start_cycle : int
    actual_start_cylce  : int
    end_cycle           : int

    depend_list         : list[DependOn] 
    is_assigned         : bool = False

    
def update_depend_on_end_cycle(node_list: list[Node], executed_node:Node) -> None:
    """ 
    Check if the node above is a dependent node for any other node
    if yes, then update DependOn.end_cycle """
    for node in node_list:

        if node.is_assigned:
            continue

        for depend in node.depend_list:
            if depend.id == executed_node.id:
                depend.end_cycle = executed_node.end_cycle

def assign_minimum_end_cycle(node_list: list[Node]) -> None:
    """ 
    For nodes that have all the dependencies assigned (with end_cycle). 
    Assign the minimum end_cycle to the minimum_start_cycle """
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

def init_node_list( graph: nx.DiGraph ) -> tuple[list, int]:
    """
    Find the dependecy node, 
    Iterate through the successor nodes and assign minimum_start_cycle (if applicable)
    Find the node that will start the earliest
    """

    print(f"\nFinding the node that will start the earliest")

    # Assign minimum_wait_time to all applicable dependent nodes
    # and finding the node that will start the earliest
    for idx, node in graph.nodes(data=True):

        if node["type"] == "dependency":

            successor_nodes         = list(graph.successors(idx))

            list_of_successor_nodes = []

            # Assign minimum_wait_time to all the successor nodes
            for successor_node in successor_nodes:
                require             = graph.edges[idx, successor_node]['weight']
                minimum_start_cycle = PACKET_SIZE * require

                # Check if the successor node has any other dependency
                dependent_list = []
                has_other_dependency = False
                predecessor_nodes    = list(graph.predecessors(successor_node))
                for predecessor_node in predecessor_nodes:
                    if predecessor_node != idx:
                        depend = DependOn(id = predecessor_node, end_cycle = None)
                        has_other_dependency = True
                        dependent_list.append(depend)

                if has_other_dependency:
                    minimum_start_cycle = None
                
                # Creating a node object
                node = Node(id = successor_node,
                            type = "task_depend",
                            minimum_start_cycle = minimum_start_cycle,
                            actual_start_cylce = None, 
                            end_cycle = None,
                            depend_list=dependent_list,
                            )

                list_of_successor_nodes.append(node)


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
    
def update_end_cylce (node_list: list[Node], node_index: int) -> Node: 
    # Assigning the end_cycle to the first executed node

    executed_node = node_list[node_index]
    processing_time = graph.nodes[executed_node.id]['processing_time']
    generate = graph.nodes[executed_node.id]['generate']
    executed_node.end_cycle = executed_node.actual_start_cylce + ( processing_time * generate ) 
    executed_node.is_assigned = True

    return executed_node

def main(graph):
    
    node_list = []

    node_list, start_node_index = init_node_list(graph)

    executed_node = update_end_cylce(node_list, start_node_index) 

    update_depend_on_end_cycle(node_list, executed_node)

    assign_minimum_end_cycle(node_list)

    # Check if all the nodes have minimum_start_cycle

    print(f"List of Successor Nodes")
    for node in node_list:
        print(node)
        
    # Condition where there is nothing in the dependecy list of the rest of the nodes.

    # Condition where there is a mix of nodes with dependency list and without dependency list

    # Loop to check if the list contains all the nodes

PACKET_SIZE = 4 # flits

if __name__ == "__main__":

    input_training_data_path    = "data/training_data/input"
    packet_list_training_path   = "data/training_data/packet_list"
    target_training_data_path   = "data/training_data/target"

    input_test_data_path        = "data/training_data/test/input"
    packet_list_test_path       = "data/training_data/test/packet_list"
    target_test_data_path       = "data/training_data/test/target"

    input_training_data_files   = natsorted(os.listdir(input_training_data_path))
    packet_list_training_files  = natsorted(os.listdir(packet_list_training_path))
    target_training_input_files = natsorted(os.listdir(target_training_data_path))

    input_test_data_files       = natsorted(os.listdir(input_test_data_path))
    packet_list_test_files      = natsorted(os.listdir(packet_list_test_path))
    target_list_test_files      = natsorted(os.listdir(target_test_data_path))

    for input_idx, target_idx, packet_list_idx in zip(input_training_data_files, target_training_input_files, packet_list_training_files):
        
        graph = load_graph_from_json(os.path.join(input_training_data_path, input_idx))

        print(f"\n--------Graph: {input_idx}--------")

        dependency_node_count = 0

        for idx, node in graph.nodes(data=True):
            if node["type"] == "dependency":
                dependency_node_count += 1

        # Skip Graphs with more than 1 dependency node
        if dependency_node_count > 1:
            continue

        compute_list    = get_compute_list_from_json(os.path.join(target_training_data_path, target_idx)) 
        packet_list     = json.load(open(os.path.join(packet_list_training_path, packet_list_idx)))

        main(graph)

        visualize_graph(graph, packet_list=packet_list, compute_list=compute_list)





        

