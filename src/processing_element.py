from src.node import Node

from src.packet import Packet
from enum import Enum


class PEStatus(Enum):
    PROCESSING = "processing"
    IDLE = "idle"
    DONE = "done"
    TRANSMITTING = "transmitting"


class ProcessingElement:
    def __init__(
        self,
        xy: tuple,
        require: int,
        generate: int,
        processing_time: int,
        packet_dest: list[int],
    ):
        self.x_y = xy

        self.require = require
        self.generate = generate
        self.packet_dest = packet_dest
        print(f"\nPacket Dest: {self.packet_dest}")

        self.processing_cycle = processing_time
        self.current_processing_cycle = 0

        self.pe_status = PEStatus.IDLE
        self.transmitting_status = PEStatus.IDLE

        self.require_count = 0
        self.generated_count = 0

        self.flits_transmitted_count = 0
        self.packet_transmitted_count = 0
        self.packet_size = 0  # in flits

    def receive_packet(self):
        """Simulate receiving a packet."""
        self.require_count += 1
        if self.require_count >= self.require:
            self.require_count = (
                self.require
            )  # Ensure it doesn't exceed the total require

    def transmit_packet(self):
        if (
            len(self.packet_dest) != self.generate
            and self.transmitting_status == PEStatus.IDLE
        ):
            raise ValueError(
                f"Length of packet_dest: {len(self.packet_dest)} != generate: {self.generate}"
            )

        if self.generated_count > 0 and self.transmitting_status == PEStatus.IDLE:
            self.transmitting_status = PEStatus.TRANSMITTING
            packet = Packet(self.x_y, self.packet_dest.pop())
            self.packet_size = packet.payload_size + packet.header_size
            print(f"Transmitting packet: {packet}, Size: {self.packet_size}")

        print(f"Look here: Transmitting status: {self.transmitting_status}")
        if self.transmitting_status == PEStatus.TRANSMITTING:
            self.flits_transmitted_count += 1
            print(f"Flits transmitted: {self.flits_transmitted_count}")
            if self.flits_transmitted_count == self.packet_size:
                print(f"Condition MET WNDAKJJSDNOAKDLKDN")
                self.transmitting_status = PEStatus.IDLE
                self.flits_transmitted_count = 0
                self.packet_transmitted_count += 1
                self.packet_size = 0

    def process(self):

        if self.transmitting_status is PEStatus.TRANSMITTING:
            self.transmit_packet()

        if self.pe_status == PEStatus.DONE or self.generated_count >= self.generate:
            self.pe_status = PEStatus.IDLE
            self.current_processing_cycle = 0
            return

        if self.require_count < self.require:
            print(f"{self.require_count}/{self.require} packets received")
            return

        if self.pe_status != PEStatus.PROCESSING:
            self.pe_status = PEStatus.PROCESSING

        self.current_processing_cycle += 1

        if self.current_processing_cycle >= self.processing_cycle:
            self.pe_status = PEStatus.DONE

            self.current_processing_cycle = 0
            self.generated_count += self.generate
            self.generated_count = min(self.generated_count, self.generate)

            if (
                self.transmitting_status is PEStatus.IDLE
                or self.transmitting_status is PEStatus.TRANSMITTING
            ):
                print(f"Init Transmitting packet")
                self.transmit_packet()
            return

    def __repr__(self):
        return f"PE({self.x_y})"


if __name__ == "__main__":

    import networkx as nx

    task_graph = nx.DiGraph()
    task_graph.add_node(0, require=0, generate=1, processing_time=10)
    task_graph.add_node(1, require=1, generate=0, processing_time=10)
    task_graph.add_edge(0, 1)

    # mapping = {task: processing_element}
    mapping_dict = {0: 16, 1: 22}
    pe_pos_dict = {16: (0, 0), 22: (2, 1)}

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

    list_of_pe = []

    for node in task_graph.nodes():
        generate = generate_labels.get(node, 0)
        require = require_labels.get(node, 0)
        processing_time = processing_time_labels.get(node, 0)

        connected_nodes = list(task_graph.successors(node))
        connected_pes = [
            pe_pos_dict[mapping_dict[connected_node]]
            for connected_node in connected_nodes
        ]

        print(
            f"Node {node} requires {require} packets and generates {generate} packets"
            f" with processing time {processing_time} cycles"
        )
        print(f"Connected Nodes: {connected_nodes}")
        print(f"Connected PEs: {connected_pes}")

        pe_id = mapping_dict[node]
        print(f"PE ID: {pe_id}")
        pe_pos = pe_pos_dict[pe_id]

        pe = ProcessingElement(
            pe_pos, require, generate, processing_time, connected_pes
        )
        list_of_pe.append(pe)

    print(f"List of Processing Elements: {list_of_pe}")

    max_cycles = 25
    print(f"\nSimulating for {max_cycles} cycles")
    for cycle in range(max_cycles):
        print(f"\nCycle: {cycle}")
        for pe in list_of_pe:
            print(f"\nProcessing Element: {pe}, Status: {pe.pe_status}")
            pe.process()
            print(
                f"Post Processing Element: {pe}, "
                f"Status: {pe.pe_status}, "
                f"Processing Cycle: {pe.current_processing_cycle}/{pe.processing_cycle}, "
                f"Generated: {pe.generated_count}/{pe.generate} packets"
            )
            print(
                f"Transmitting Status: {pe.transmitting_status}, "
                f"Flites Transmitted: {pe.flits_transmitted_count}/{pe.packet_size}, "
                f"Packet Transmitted: {pe.packet_transmitted_count}/{pe.generate}"
            )
