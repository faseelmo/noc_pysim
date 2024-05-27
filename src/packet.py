from src.processing_element import ProcessingElement
from src.router import Router

from typing import Union

class Packet: 
    def __init__(self, bytes: int, 
                 source : Union[Router, ProcessingElement], 
                 destination: Union[Router, ProcessingElement], 
                 routing_links: list):

        self.size = bytes
        self.source = source
        self.destination = destination
        self.routing_links = routing_links
        self.current_link = None
        self.current_node = None

    def set_current_node(self, node):
        self.current_node = node

    def set_current_link(self, link):
        self.current_link = link

    def pop_routing_link(self):
        return self.routing_links.pop(0)

    def has_more_routing_links(self):
        return len(self.routing_links) > 0

    def __repr__(self):
        return f"Packet({self.size} Bytes) from {self.source} to {self.destination} at {self.current_node}"

if __name__ == "__main__":

    from src.mesh_network import MeshNetwork
    mesh = MeshNetwork()
    src = mesh.get_processing_element(0, 0)
    dest = mesh.get_processing_element(1, 1)
    routing_links = mesh.get_routing_links(src, dest)

    packet = Packet(10, src, dest, routing_links)
    print(packet.routing_links)
    print(packet)