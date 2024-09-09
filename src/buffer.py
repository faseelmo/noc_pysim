from enum           import Enum
from typing         import Union
from collections    import deque

class BufferStatus(Enum):
    EMPTY       = "empty"
    AVAILABLE   = "available"
    FULL        = "full"

class Buffer:
    def __init__(self, size: int):
        self.size   = size
        self.queue  = deque(maxlen=size)
        self.status = BufferStatus.EMPTY

    def add_flit(self, flit: Union[dict, int]) -> None:
        """
        Adds flit to the buffer if there is space available.
        Updates the status of the buffer.
        State transitions:
            EMPTY      -> AVAILABLE, for the first flit.
            AVAILABLE  -> FULL, when the buffer is full.
        
        """
        if self.status in (BufferStatus.EMPTY, BufferStatus.AVAILABLE ): # Basically or condition

            self.queue.append(flit)

            if self.status == BufferStatus.EMPTY:
                self.status = BufferStatus.AVAILABLE

            if len(self.queue) == self.size:
                self.status = BufferStatus.FULL

        else:
            raise Exception("Cannot add to full buffer")

    def remove(self) -> Union[dict, int]:
        """
        Removes flits from the buffer if there are any.
        Updates the status of the buffer.
        returns the flit that was removed.
        State transitions:
            FULL        -> AVAILABLE, when the buffer is not full.
            AVAILABLE   -> EMPTY, when the buffer is empty.
        """

        if self.status == BufferStatus.EMPTY:
            raise Exception("Cannot remove from empty buffer")

        if self.status in (BufferStatus.FULL, BufferStatus.AVAILABLE):

            flit = self.queue.popleft()

            if len(self.queue) == 0:
                self.status = BufferStatus.EMPTY

            if self.status == BufferStatus.FULL:
                self.status = BufferStatus.AVAILABLE

            return flit

    def __str__(self):
        return f"{self.status} -> {list(self.queue)}"


if __name__ == "__main__":

    from .packet import Packet

    buffer = Buffer(4)
    print(f"\n{buffer}")

    packet = Packet(source_xy=(0, 0), 
                    dest_xy=(1, 1), 
                    source_task_id=0)

    packet_is_transmitted = False
    while not packet_is_transmitted:
        packet_is_transmitted, flit = packet.transmit_flit()
        buffer.add_flit(flit)
        print(f"{buffer}")

    print(f"\nRemoving\n{buffer}")
    for i in range(4):
        buffer.remove()
        print(f"{buffer}")


