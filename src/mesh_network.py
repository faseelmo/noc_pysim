# from router import Router
from src.processing_element import ProcessingElement
from src.router import Router
from src.link import Link

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

    def route_packet(self, source: ProcessingElement, destination: ProcessingElement) -> None:
        routing_list = []
        pass

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

if __name__ == "__main__":
    mesh = MeshNetwork()

    r1 = mesh.get_router(0, 0)
    pe1 = mesh.get_processing_element(0, 0)
    print(f"Router 1: {r1}")
    print(f"Processing Element 1: {pe1}")

    r2 = mesh.get_router(1, 0)
    pe2 = mesh.get_processing_element(1, 0)
    print(f"Router 2: {r2}")
    print(f"Processing Element 2: {pe2}")

    link1 = mesh.get_link(r1, r2)
    print(f"Link 1: {link1}")


