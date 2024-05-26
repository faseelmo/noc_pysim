import math 

class Node: 
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.status = 'idle'
        self.cycles_required_for_next_hop = 0

    def send_packet(self, packet, current_cycle: int):

        from src.packet import Packet
        if not isinstance(packet, Packet):
            raise TypeError("packet must be an instance of Packet()")

        if self.status == 'idle': 
            packet.current_node = self
            packet.current_link = packet.routing_links.pop(0)
            self.status = 'transmitting'

            self.cycles_required_for_next_hop = math.ceil(packet.size / packet.current_link.bandwidth)
            print(f"Cycles require for next hop is {self.cycles_required_for_next_hop}")

        packet.current_link.transmit(current_cycle, self.cycles_required_for_next_hop)
        packet.current_link.check_transmission(current_cycle)

        if packet.current_link.is_busy == False:
            self.status = 'done'
            dest_node = packet.current_link.get_dest_node(self)
            print(f"Packet sent to {dest_node}")
            print(f"Packet is {packet}")
            packet.current_node = dest_node
            packet.current_link = None
            print(f"Packet After is {packet}")

            if len(packet.routing_links) == 0:
                self.status = 'recieved'