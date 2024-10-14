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


def draw_router_status(router, color_map, ax): 
    from src.flit import HeaderFlit, PayloadFlit, TailFlit, EmptyFlit
    import matplotlib.pyplot as plt
    
    # Define buffer names and coordinates for layout (directions)
    buffer_layout = {
        "Local": (0, 0),
        "West": (-1, 0),
        "North": (0, 1),
        "East": (1, 0),
        "South": (0, -1)
    }
    
    # Define buffer directions for input/output
    buffer_directions = {
        "local_input": router._local_input_buffer,
        "local_output": router._local_output_buffer,
        "west_input": router._west_input_buffer,
        "west_output": router._west_output_buffer,
        "north_input": router._north_input_buffer,
        "north_output": router._north_output_buffer,
        "east_input": router._east_input_buffer,
        "east_output": router._east_output_buffer,
        "south_input": router._south_input_buffer,
        "south_output": router._south_output_buffer,
    }
    
    
    # Draw each buffer's status
    for buffer_name, buffer_obj in buffer_directions.items():
        # Determine position and type of buffer (input/output)
        direction = buffer_name.split('_')[0].capitalize()
        buf_type = buffer_name.split('_')[1].capitalize()
        position = buffer_layout[direction]

        if buf_type == "Input":
            position = (position[0], position[1] - 0.1)
        elif buf_type == "Output": 
            position = (position[0], position[1] + 0.1)

        # Position the buffer label with an offset
        label_position = (position[0], position[1]) 

        # Create a visual block for each buffer (without overlapping with circles)
        ax.text(label_position[0], label_position[1], f"{direction} {buf_type}", 
                ha='center', va='center', fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black'))

        # Draw the flits in the buffer
        for idx, flit in enumerate(buffer_obj.queue):
            if isinstance(flit, EmptyFlit):
                continue
            
            if isinstance(flit, HeaderFlit):
                label = 'H'
            elif isinstance(flit, PayloadFlit):
                label = 'P'
            elif isinstance(flit, TailFlit):
                label = 'T'
            else:
                label = '?'

            # Assign a unique color based on the packet UID
            packet_uid = flit.get_uid()
            if packet_uid not in color_map:
                color_map[packet_uid] = (random.random(), random.random(), random.random())
            color = color_map[packet_uid]

            # Plot the flit as a circle with label
            if buf_type == "Input":
                flit_y_offset = -0.2
            elif buf_type == "Output":
                flit_y_offset = 0.2

            ax.add_patch(plt.Circle((position[0] + 0.1 * idx, position[1] + flit_y_offset), 0.08, color=color, ec='black'))
            ax.text(position[0] + 0.1 * idx, position[1] + flit_y_offset, label, fontsize=12, ha='center', va='center', color='white')
    
    # Set the plot limits and title
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title(f"Router ({router._x}, {router._y})")
    
    ax.axis('off')

    return color_map