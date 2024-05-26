import math
from src.node import Node

class ProcessingElement(Node):
    def __init__(self, x, y):
        super().__init__(x, y)

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
            src.status = 'idle'
            break



    

