from src.router import Router
from src.processing_element import ProcessingElement

class Link:
    def __init__(self, source, destination):
        
        self.nodes = frozenset([source, destination])
        self.source = source
        self.destination = destination

        if isinstance(source, Router) and isinstance(destination, Router):
            self.link_type = 'router_link'
        elif isinstance(source, Router) and isinstance(destination, ProcessingElement ):
            self.link_type = 'pe_link'
        else:  
            raise ValueError("Invalid link type")

    def __repr__(self):
        return f"{self.source} - {self.destination}"

    def __eq__(self, other):
        if isinstance(other, Link):
            return self.nodes == other.nodes
        return False

    def __hash__(self):
        return hash(self.nodes)

if __name__ == "__main__":
    Link()