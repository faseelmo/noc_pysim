from src.router import Router
from src.processing_element import ProcessingElement

from typing import Union

class Link:
    def __init__(self, node1: Union[Router, ProcessingElement] , node2: Union[Router, ProcessingElement]):

        self.nodes = frozenset([node1, node2])
        self.bandwidth =  10 # bytes per cycle 
        self.is_busy = False
        self.transmission_end_cycle = 0

        if isinstance(node1, Router) and isinstance(node2, Router):
            self.link_type = 'router_link'
        elif (isinstance(node1, Router) and isinstance(node2, ProcessingElement)) or \
             (isinstance(node2, Router) and isinstance(node1, ProcessingElement)):
            self.link_type = 'pe_link'
        else:  
            raise ValueError("Invalid link type")

    def transmit(self, cycles_required, current_cycle): 
        if self.is_busy:
            print(f"{self} is busy")
            return 

        self.is_busy = True
        self.transmission_end_cycle = current_cycle + cycles_required - 1 # -1 because current_cycle is included
        print(f"Transmission will end at {self.transmission_end_cycle}")

    def get_dest_node(self, src_node): 
        for node in self.nodes:
            if node != src_node:
                return node 
        raise ValueError("src_node node in link")

    def check_transmission(self, current_cycle):
        """Returns True if transmission is still ongoing, False otherwise"""
        print(f"Current Cycle is {current_cycle}")
        if self.is_busy and self.transmission_end_cycle == current_cycle: 
            print(f"Transmission Completed")
            self.is_busy = False
        return self.is_busy
        

    def __repr__(self): 
        nodes_list = list(self.nodes)
        return f"Link({nodes_list[0]} - {nodes_list[1]})"

    def __eq__(self, other):
        if isinstance(other, Link):
            return self.nodes == other.nodes
        return False

    def __hash__(self):
        return hash(self.nodes)

if __name__ == "__main__":
    print(Link(Router(0, 0), Router(0, 1)))