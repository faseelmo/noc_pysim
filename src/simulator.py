from src.mesh_network import MeshNetwork
from src.packet import Packet
# from src.router import Router   
# from src.processing_element import ProcessingElement

class Simulator: 
    def __init__(self, max_cycles): 
        self.max_cycles = max_cycles
        self.network = MeshNetwork()

    def run(self):
        src_pe = self.network.get_processing_element(0, 0)
        dest_pe = self.network.get_processing_element(1, 1)
        list_of_links = self.network.get_routing_links(src_pe, dest_pe)

        packet = Packet(100)
        communication_link = {}

        for idx, link in enumerate(list_of_links):
            communication_link['idx'] = idx 
            communication_link['link'] = link
            communication_link['status'] = 'scheduled'
            communication_link['packet'] = packet

        for key, value in communication_link.items():
            print(f"{key} : {value}")

        for cycle in range(self.max_cycles):
            print(f"Current cycle is {cycle}")
            current_cycle = cycle
            self.network.simulate_cycle(current_cycle, communication_link)

if __name__ == "__main__":

    sim = Simulator(150)
    sim.run()