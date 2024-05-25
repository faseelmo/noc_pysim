from src.router import Router
from src.processing_element import ProcessingElement

from typing import Union

class Link:
    def __init__(self, source: Union[Router, ProcessingElement] , destination: Union[Router, ProcessingElement]):

        self.nodes = frozenset([source, destination])
        self.source = source
        self.destination = destination
        self.bandwidth =  10 # bytes per cycle 
        self.is_busy = False
        self.transmission_end_cycle = 0

        if isinstance(source, Router) and isinstance(destination, Router):
            self.link_type = 'router_link'
        elif isinstance(source, Router) and isinstance(destination, ProcessingElement ):
            self.link_type = 'pe_link'
        else:  
            raise ValueError("Invalid link type")

    def transmit(self, cycles_required, current_cycle): 
        if self.is_busy:
            print(f"Link is busy")
            return 

        print(f"\nStarting Transmission")
        self.is_busy = True
        self.transmission_end_cycle = current_cycle + cycles_required
        print(f"Transmission will end at {self.transmission_end_cycle}")

    def check_transmission(self, current_cycle):

        if self.is_busy and self.transmission_end_cycle == current_cycle: 
            print(f"Transmission Completed")
            self.is_busy = False

    def __repr__(self): 
        return f"{self.source} - {self.destination}"

    def __eq__(self, other):
        if isinstance(other, Link):
            return self.nodes == other.nodes
        return False

    def __hash__(self):
        return hash(self.nodes)

if __name__ == "__main__":
    print(Link(Router(0, 0), Router(0, 1)))