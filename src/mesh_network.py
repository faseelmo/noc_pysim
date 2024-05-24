# from router import Router
from src.processing_element import ProcessingElement
from src.router import Router
from src.link import Link

import matplotlib.pyplot as plt

class MeshNetwork:
    def __init__(self): 
        
        self.size                   =   4
        self.routers                =   [[Router(x,y) for y in range(self.size)] for x in range(self.size)]
        self.processing_elements    =   [[ProcessingElement(x,y) for y in range(self.size)] for x in range(self.size)]
        self.links                  =   self.create_links()

        print(f"Links created: {len(self.links)}")
        for key,value in self.links.items():
            print(f"{value}")

    def create_links(self):
        link_dict = {}
        for x in range(self.size):
            for y in range(self.size): 
                if x  > 0: # connect to the left
                    key = frozenset({(x, y), (x-1, y)})
                    if key not in link_dict:
                        link_dict[key] = Link(self.routers[x][y], self.routers[x-1][y])
                if x < self.size - 1: # connect to the right
                    key = frozenset({(x, y), (x+1, y)})
                    if key not in link_dict:
                        link_dict[key] = Link(self.routers[x][y], self.routers[x+1][y])
                if y > 0: # connect to the top
                    key = frozenset({(x, y), (x, y-1)})
                    if key not in link_dict:
                        link_dict[key] = Link(self.routers[x][y], self.routers[x][y-1])
                if y < self.size - 1: # connect to the bottom
                    key = frozenset({(x, y), (x, y+1)})
                    if key not in link_dict:
                        link_dict[key] = Link(self.routers[x][y], self.routers[x][y+1])

                key = frozenset({(x, y), (x, y)})
                if key not in link_dict:
                    link_dict[key] = Link(self.routers[x][y], self.processing_elements[x][y])

        return link_dict

    def get_link(self, source, destination):
        return self.links.get(frozenset({source, destination}))

if __name__ == "__main__":
    mesh = MeshNetwork()
