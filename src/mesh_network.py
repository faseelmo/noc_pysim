from src.link import Link
from src.packet import Packet
from src.router import Router
from src.processing_element import ProcessingElement

from typing import Union

class MeshNetwork:
    def __init__(self): 
        self.size = 4
        self.routers = {(x, y): Router(x, y) for x in range(self.size) for y in range(self.size)}
        self.processing_elements = {(x, y): ProcessingElement(x, y) for x in range(self.size) for y in range(self.size)}
        self.links = self.create_links()

        print(f"Length of routers: {len(self.routers)}")
        print(f"Length of processing elements: {len(self.processing_elements)}")
        print(f"Length of links: {len(self.links)}")

    def create_links(self) -> dict:
        link_dict = {}
        for x in range(self.size):
            for y in range(self.size): 
                if x  > 0: # connect to the left
                    key = frozenset({(x, y), (x-1, y)})
                    if key not in link_dict:
                        link_dict[key] = Link(self.routers[(x, y)], self.routers[(x-1, y)])
                if x < self.size - 1: # connect to the right
                    key = frozenset({(x, y), (x+1, y)})
                    if key not in link_dict:
                        link_dict[key] = Link(self.routers[(x, y)], self.routers[(x+1, y)])
                if y > 0: # connect to the top
                    key = frozenset({(x, y), (x, y-1)})
                    if key not in link_dict:
                        link_dict[key] = Link(self.routers[(x, y)], self.routers[(x, y-1)])
                if y < self.size - 1: # connect to the bottom
                    key = frozenset({(x, y), (x, y+1)})
                    if key not in link_dict:
                        link_dict[key] = Link(self.routers[(x, y)], self.routers[(x, y+1)])
                key = frozenset({(x, y), (x, y)})
                if key not in link_dict:
                    link_dict[key] = Link(self.routers[(x, y)], self.processing_elements[(x, y)])
        return link_dict

    def get_routing_links(self, source: ProcessingElement, destination: ProcessingElement) -> list:
        routing_link_list = []

        x_src, y_src = source.x, source.y
        x_dest, y_dest = destination.x, destination.y

        """if src is on the left of dest, step_x is 1, else -1"""
        step_x = 1 if x_src < x_dest else -1 
        src_pe = self.get_processing_element(x_src, y_src)
        src_router = self.get_router(x_src, y_src)
        link = self.get_link(src_pe, src_router)

        routing_link_list.append(link)

        while x_src != x_dest:
            current_router = self.get_router(x_src, y_src)
            x_src += step_x
            next_router = self.get_router(x_src, y_src)
            link = self.get_link(current_router, next_router)
            routing_link_list.append(link)
        # print(f"Transversing in x-direction done")

        """if src is below dest, step_y is 1 (i.e we want to move up), else -1"""
        step_y = 1 if y_src < y_dest else -1
        while y_src != y_dest:
            current_router = self.get_router(x_src, y_src)
            y_src += step_y
            next_router = self.get_router(x_src, y_src)
            link = self.get_link(current_router, next_router)
            routing_link_list.append(link)
        # print(f"Transversing in y-direction done")

        dest_pe = self.get_processing_element(x_dest, y_dest)
        dest_router = self.get_router(x_dest, y_dest)
        link = self.get_link(dest_router, dest_pe)
        routing_link_list.append(link)
        return routing_link_list

    def get_router(self, x: int, y: int) -> Router:
        return self.routers.get((x, y))

    def get_processing_element(self, x: int, y: int) -> ProcessingElement:
        return self.processing_elements.get((x, y))

    def get_link(self, source: Union[Router, ProcessingElement], destination: Union[Router, ProcessingElement]) -> Link:
        source_xy = (source.x, source.y)
        destination_xy = (destination.x, destination.y)
        link = self.links.get(frozenset({source_xy, destination_xy}))
        assert link is not None, f"Link between {source} and {destination} does not exist"
        return link

    def visualize_mesh(self): 
        import networkx as nx


if __name__ == "__main__":
    mesh = MeshNetwork()

    def test_get_routing_links(pe1_xy, pe2_xy):
        print(f"")
        pe1 = mesh.get_processing_element(*pe1_xy)
        pe2 = mesh.get_processing_element(*pe2_xy)
        print(f"{pe1} -> {pe2}")
        routing = mesh.get_routing_links(pe1, pe2)
        for link in routing:
            print(link)

    """Routing Tests"""
    # bottom left to top right 
    test_get_routing_links((0, 0), (3, 3))

    # top right to bottom left
    test_get_routing_links((3, 3), (0, 0))

    # top left to bottom right
    test_get_routing_links((0, 3), (3, 0))

    # bottom right to top left
    test_get_routing_links((3, 0), (0, 3))


    def test_simulate_cycle(pe1_xy, pe2_xy, max_cycles):
        from src.node import NodeStatus
        pe1 = mesh.get_processing_element(*pe1_xy)
        pe2 = mesh.get_processing_element(*pe2_xy)

        routing = mesh.get_routing_links(pe1, pe2)
        packet = Packet(70, pe1, pe2, routing)

        print(f"\nPacket Routing Before cycle starting is {routing}")

        node = pe1
        for cycle in range(max_cycles):
            if node.status == NodeStatus.PACKET_ARRIVED_DEST:
                print(f"Packet has been recieved by {node}")
                break

            if node.status != NodeStatus.TX_DONE:
                node.send_packet(packet, cycle)

            elif node.status == NodeStatus.TX_DONE:
                node.status = NodeStatus.IDLE
                node = packet.current_node

        print(f"\nPacket Routing after cycle is {routing}")

    # test_simulate_cycle((0, 0), (2, 2), 100)
    # test_simulate_cycle((1, 0), (2, 0), 100)


    def test_multiple_packets():
        packet_1_src = mesh.get_processing_element(0, 0)
        packet_1_dest = mesh.get_processing_element(2, 0)
        routing_1 = mesh.get_routing_links(packet_1_src, packet_1_dest)
        packet_1 = Packet(30, packet_1_src, packet_1_dest, routing_1, 1)

        packet_2_src = mesh.get_processing_element(1, 0)
        packet_2_dest = mesh.get_processing_element(2, 0)
        routing_2 = mesh.get_routing_links(packet_2_src, packet_2_dest)
        packet_2 = Packet(30, packet_2_src, packet_2_dest, routing_2, 2)

        nodes = [packet_1_src, packet_2_src] # initial nodes
        packets = [packet_1, packet_2] 

        from src.node import NodeStatus
        for cycle in range(100):
            print(f"\nCycle {cycle}")
            print(f"Status, {nodes[0]}:{nodes[0].status}, {nodes[1]}:{nodes[1].status}")
            for i in range(len(nodes)):
                node = nodes[i]
                packet = packets[i]
                if node.status == NodeStatus.PACKET_ARRIVED_DEST:
                    continue
                
                if node.status != NodeStatus.TX_DONE:
                    node.send_packet(packet, cycle)
    
                elif node.status == NodeStatus.TX_DONE:
                    node.status = NodeStatus.IDLE
                    nodes[i] = packet.current_node
    
            if all(node.status == NodeStatus.PACKET_ARRIVED_DEST for node in nodes):
                print(f"All Packets have been recieved")
                print(f"Packet 1 at {nodes[0].status}")
                print(f"Packet 2 at {nodes[1].status}")
                break
            
           
    print(f"\n\n--- Multi packet ---\n\n")
    test_multiple_packets()
        # print(f"\nPacket Routing after cycle is {routing}")

