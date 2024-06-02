from packet import Packet
from enum import Enum


class PEStatus(Enum):
    PROCESSING = "processing"
    IDLE = "idle"
    DONE = "done"


class TXStatus(Enum):
    IDLE = "idle"
    DONE = "done"
    TRANSMITTING = "transmitting"


class ProcessingElement:
    def __init__(
        self,
        xy: tuple[int, int],
        require: int,
        generate: int,
        processing_time: int,
        packet_dest_list: list[dict],
    ):
        self.x_y = xy

        self.require = require
        self.generate = generate
        self.packet_dest_list = packet_dest_list

        self.current_processing_cycle = 0
        self.processing_cycle = processing_time

        self.pe_status = PEStatus.IDLE
        self.transmitting_status = TXStatus.IDLE

        self.require_count = 0
        self.generated_count = 0

        self.packet_size_in_flits = 0
        self.flits_transmitted_count = 0
        self.packet_transmitted_count = 0

        self.transmit_require = 0
        self.transmit_generate = 0
        self.current_transmit_packet = None

    def receive_packet(self):
        """Simulate receiving a packet."""
        self.require_count += 1

    def validate_require_count(self):
        # In finalize_processing(), generated_count is incremented
        total_require_count = sum(
            packet_dest["require"] for packet_dest in self.packet_dest_list
        )
        if total_require_count != (self.generate - self.generated_count + 1):
            raise ValueError(
                f"Require needed: {total_require_count} != generate left: {(self.generate - self.generated_count)}"
            )

    def initiate_packet_transmission(self):
        if self.transmit_generate == self.transmit_require:
            print(f"First Transmission")
            self.transmit_generate = 0  # Resetting for new destination
            self.transmitting_status = TXStatus.TRANSMITTING
            packet_dest = self.packet_dest_list.pop()
            self.current_transmit_packet = Packet(self.x_y, packet_dest["pos"])
            self.transmit_require = packet_dest["require"]
            self.packet_size_in_flits = self.current_transmit_packet.size

        elif self.transmit_generate < self.transmit_require:
            print(f"Second Transmission")
            self.transmitting_status = TXStatus.TRANSMITTING
            self.transmit_require -= 1  # Dec for the second tx to the same dest
            self.packet_size_in_flits = self.current_transmit_packet.size

        else:
            raise ValueError(f"Error in initiating packet transmission")

        print(
            f"\tTransmitting packet: {self.current_transmit_packet}, "
            f"require: {self.transmit_require}, "
            f"Size: {self.packet_size_in_flits}\n"
        )
        return

    def complete_packet_transmission(self):
        self.transmitting_status = TXStatus.IDLE

        self.flits_transmitted_count = 0
        self.packet_transmitted_count += 1
        self.packet_size_in_flits = 0

        self.transmit_generate += 1

    def transmit_packet(self):
        if (
            self.generate == self.generated_count
            and self.transmitting_status is TXStatus.IDLE
        ):
            if self.transmit_generate < self.transmit_require:
                self.initiate_packet_transmission()
            return  # Check position of return

        if self.generated_count > 0 and self.transmitting_status is TXStatus.IDLE:
            self.validate_require_count()
            self.initiate_packet_transmission()

        if self.transmitting_status is TXStatus.TRANSMITTING:
            self.flits_transmitted_count += 1
            if self.flits_transmitted_count == self.packet_size_in_flits:
                self.complete_packet_transmission()

    def reset_processing_status(self):
        self.pe_status = PEStatus.IDLE
        self.current_processing_cycle = 0
        return

    def complete_processing(self):
        self.pe_status = PEStatus.DONE
        self.current_processing_cycle = 0
        self.generated_count += 1
        if (
            self.transmitting_status is TXStatus.IDLE
            or self.transmitting_status is TXStatus.TRANSMITTING
        ):
            self.transmit_packet()
        return

    def process(self):

        if self.transmitting_status is TXStatus.TRANSMITTING:
            self.transmit_packet()

        if self.pe_status == PEStatus.DONE or self.generated_count >= self.generate:
            self.reset_processing_status()

        if self.pe_status != PEStatus.PROCESSING and self.require_count == self.require:
            self.pe_status = PEStatus.PROCESSING

        self.current_processing_cycle += 1

        if self.current_processing_cycle >= self.processing_cycle:
            self.complete_processing()

    def __repr__(self):
        return f"PE({self.x_y})"


if __name__ == "__main__":

    import networkx as nx

    task_graph = nx.DiGraph()
    task_graph.add_node(0, require=0, generate=3, processing_time=10)
    task_graph.add_node(1, require=2, generate=0, processing_time=10)
    task_graph.add_node(2, require=1, generate=0, processing_time=10)
    task_graph.add_edge(0, 1)
    task_graph.add_edge(0, 2)

    # mapping = {task: processing_element}
    mapping_dict = {0: 16, 1: 22, 2: 19}
    pe_pos_dict = {16: (0, 0), 22: (2, 1), 19: (3, 0)}

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

    import matplotlib.pyplot as plt

    plt.show()

    list_of_pe = []

    for node in task_graph.nodes():
        generate = generate_labels.get(node, 0)
        require = require_labels.get(node, 0)
        processing_time = processing_time_labels.get(node, 0)

        connected_nodes = list(task_graph.successors(node))

        connected_pe_list = []
        for connected_node in connected_nodes:
            connected_pes_dict = {}
            connected_pes_dict["pe"] = connected_node
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
        list_of_pe.append(pe)

    print(f"List of Processing Elements: {list_of_pe}")

    max_cycles = 35
    print(f"\nSimulating for {max_cycles} cycles")
    for cycle in range(max_cycles):
        print(f"\n----Cycle: {cycle}----")
        for pe in list_of_pe:

            print(f"\n{pe}")
            print(f"PRE:\t {pe.pe_status}")
            pe.process()
            print(
                f"POST:\t {pe.pe_status}, "
                f"Processing Cycle: {pe.current_processing_cycle}/{pe.processing_cycle - 1}, "
                f"Generated: {pe.generated_count}/{pe.generate} packets"
            )
            print(
                f"Tx:\t {pe.transmitting_status}, "
                f"Flites Transmitted: {pe.flits_transmitted_count}/{pe.packet_size_in_flits}, "
                f"Packet Transmitted: {pe.packet_transmitted_count}/{pe.generate}"
            )
