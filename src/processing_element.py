import math

class ProcessingElement: 
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

            dest_node = packet.current_link.get_dest_node(self)
            print(f"Dest node is {dest_node}")
        
            self.cycles_required_for_next_hop = math.ceil(packet.size / packet.current_link.bandwidth)
            print(f"Cycles require for next hop is {self.cycles_required_for_next_hop}")

        packet.current_link.transmit(current_cycle, self.cycles_required_for_next_hop)
        packet.current_link.check_transmission(current_cycle)

        if packet.current_link.is_busy == False:
            self.status = 'done'




        pass

    def __repr__(self):
        return f"PE({self.x}, {self.y})"


if __name__ == "__main__":
    

    from src.mesh_network import MeshNetwork
    mesh = MeshNetwork()
    src = mesh.get_processing_element(0, 0)     # Type ProcessingElement()
    dest = mesh.get_processing_element(1, 1)    # Type ProcessingElement()
    routing_links = mesh.get_routing_links(src, dest)

    from src.packet import Packet
    packet = Packet(40, src, dest, routing_links)
    print(f"\nPacket Routing is ")
    print(packet.routing_links)
    print(packet)

    src.send_packet(packet, 1)

    for cycle in range(100): 
        if src.status != 'done':
            src.send_packet(packet, cycle)
        elif src.status == 'done' : 
            print(f"Packet sent")
            break



    

