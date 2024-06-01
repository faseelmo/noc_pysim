import math 

from enum import Enum

class NodeStatus(Enum):
    IDLE = 'idle'
    TRANSMITTING = 'transmitting'
    TX_DONE = 'tx_done'
    PACKET_ARRIVED_DEST = 'packet_arrived_dest'

class Node: 
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.status = NodeStatus.IDLE 
        self.cycles_required_for_next_hop = 0

    def send_packet(self, packet, current_cycle: int):
        from src.packet import Packet
        if not isinstance(packet, Packet):
            raise TypeError("packet must be an instance of Packet()")

        if self.status == NodeStatus.IDLE: 
            packet.set_current_node(self)
            packet.set_current_link(packet.pop_routing_link())
            self.status = NodeStatus.TRANSMITTING
            print(f"1. {self} status changed to {self.status}")
            self.cycles_required_for_next_hop = math.ceil(packet.size / packet.current_link.bandwidth)
            print(f"\nCycles require for next hop is {self.cycles_required_for_next_hop}")

        packet.current_link.transmit(current_cycle, self.cycles_required_for_next_hop, packet)
        link_is_busy = packet.current_link.check_transmission(current_cycle)

        if  not link_is_busy:
            self.status = NodeStatus.TX_DONE
            print(f"2. {self} status changed to {self.status}")
            dest_node = packet.current_link.get_dest_node(self)
            packet.current_node = dest_node
            packet.current_link = None

            if not packet.has_more_routing_links():
                print(f"{packet} has arrived at final destination {dest_node}")
                print(f"3. {self} status changed to {self.status}")
                self.status = NodeStatus.PACKET_ARRIVED_DEST