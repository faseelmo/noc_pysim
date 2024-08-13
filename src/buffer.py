from collections import deque


class Buffer:
    def __init__(self, size: int):
        self.size = size
        self.queue = deque(maxlen=size)

    def add(self, flit) -> None:
        if len(self.queue) < self.size:
            self.queue.append(flit)
        else:
            print("Buffer is full")

    def add_packet(self, packet) -> None:
        if len(self.queue) + packet.size <= self.size:
            self.queue.append(packet.header_info)
            for i in range(packet.payload_size):
                self.queue.append(i)
        else:
            print(f"Not enough space in buffer to add packet")

    def remove(self):
        if len(self.queue) > 0:
            return self.queue.popleft()
        else:
            print("Buffer is empty")
            return 

    def __str__(self):
        return f"Buffer: {self.queue}"


if __name__ == "__main__":
    input_buffer = Buffer(4)

    from packet import Packet

    packet = Packet((0, 0), (1, 1))

    input_buffer.add(packet.header_info)
    for i in range(packet.payload_size):
        input_buffer.add(i)

    input_buffer.add(None)

    print(f"Buffer is {input_buffer}")
    input_buffer.remove()
    print(f"Buffer after removing is {input_buffer} ")

    new_buffer = Buffer(4)
    new_buffer.add_packet(packet)
    print(f"\nBuffer after adding packet is {new_buffer}")
    new_buffer.remove()
    print(f"Buffer after removing is {new_buffer}")
