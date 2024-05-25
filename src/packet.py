from src.link import Link

class Packet: 
    def __init__(self, bytes: int):
        # self.link = link
        self.size = bytes

    def __repr__(self):
        return f"Packet({self.size})"

if __name__ == "__main__":
    from src.router import Router
    link = Link(Router(0, 0), Router(0, 1))
    packet = Packet(link, 10)
    print(packet)