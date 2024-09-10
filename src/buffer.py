from enum           import Enum
from typing         import Union
from collections    import deque

from .flit import HeaderFlit, PayloadFlit, TailFlit

class BufferStatus(Enum):
    EMPTY       = "empty"
    AVAILABLE   = "available"
    FULL        = "full"

class Buffer:
    def __init__(self, size: int):
        self.size   = size
        self.queue  = deque(maxlen=size)
        self.status = BufferStatus.EMPTY

    def add_flit(self, flit: Union[HeaderFlit, PayloadFlit, TailFlit]) -> None:
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

    def remove(self) -> Union[HeaderFlit, PayloadFlit, TailFlit, None]:
        """
        Removes flits from the buffer if there are any.
        - Updates the status of the buffer.
        Returns the flit that was removed or if the buffer is not fully populated, 
            returns False.
        State transitions:
            FULL        -> AVAILABLE,   when the buffer is not full.
            AVAILABLE   -> EMPTY,       when the buffer is empty.
        """

        if self.status == BufferStatus.EMPTY:
            raise Exception("Cannot remove from empty buffer")

        if self.status == BufferStatus.AVAILABLE:
            # Last element in the buffer is empty. Returns False.
            return None


        if self.status in (BufferStatus.FULL, BufferStatus.AVAILABLE):

            flit = self.queue.popleft()

            if len(self.queue) == 0:
                self.status = BufferStatus.EMPTY

            if self.status == BufferStatus.FULL:
                self.status = BufferStatus.AVAILABLE

            return flit

    def __str__(self):
        queue_str = [str(item) for item in self.queue]
        return f"{self.status} -> {queue_str}"


if __name__ == "__main__":

    from .packet import Packet

    buffer = Buffer(4)
    print(f"\n{buffer}")

    """
    Test 1: Adding 4 flits to the buffer and then removing them.
    """
    print( "\n- Test 1: Adding 4 flits to the buffer and then removing them." )
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

    """
    Test 2: Adding 2 flits to the buffer and then removing them.
    """
    print( "\n- Test 2: Adding 2 flits to the buffer and then removing them." )
    
    packet = Packet(source_xy=(0, 0), 
                    dest_xy=(1, 1), 
                    source_task_id=0)

    buffer = Buffer(4)

    for i in range(2):
        packet_is_transmitted, flit = packet.transmit_flit()
        buffer.add_flit(flit)
        print(f"{buffer}")

    print( f"Buffer after adding 2 flits: {buffer}" )

    flit = buffer.remove()
    print( f"Buffer after removing 1 flit: {buffer}, removed flit {flit}" )




