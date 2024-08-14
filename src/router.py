from buffer import Buffer

from enum import Enum


class NIStatus(Enum):
    FULL = "full"
    NOT_FULL = "not_full"


class Router:
    def __init__(self, x_y: tuple, buffer_size: int):
        self.x = x_y[0]
        self.y = x_y[1]

        self.network_interface = Buffer(buffer_size)

        self.west_input_buffer = Buffer(buffer_size)
        self.north_input_buffer = Buffer(buffer_size)
        self.east_input_buffer = Buffer(buffer_size)
        self.south_input_buffer = Buffer(buffer_size)

    # def process()

    def do_routing(self):
        pass

    def check_network_interface_status(self):
        if len(self.network_interface.queue) == self.network_interface.size:
            return NIStatus.FULL
        else:
            return NIStatus.NOT_FULL

    def add_packet_to_network_interface(self, packet):
        print(f"{self} Adding packet to network interface")
        self.network_interface.add_packet(packet)

    def __repr__(self):
        return f"R({self.x}, {self.y})"


if __name__ == "__main__":
    router = Router()
