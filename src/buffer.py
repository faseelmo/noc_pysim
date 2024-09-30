from enum           import Enum
from typing         import Union
from collections    import deque

from .flit import HeaderFlit, PayloadFlit, TailFlit, EmptyFlit

class Buffer:
    def __init__(self, size: int):
        self.size               = size
        self.queue              = deque(maxlen=size)
        self._in_transmit_mode   = False
        self.fill_with_empty_flits()

    def add_flit(self, flit: Union[HeaderFlit, PayloadFlit, TailFlit]) -> None:
        """
        Adds flit to the buffer if it is not full.
        Full is defined as having all non-empty flits.
        """
        if self.is_full(): 
            raise Exception("Cannot add to full buffer")
        else: 
            self.queue.append(flit)

        # print(f"Buffer after adding flit")
        # for flit in self.queue:
        #     print(f"{flit}")

    def peek(self) -> Union[HeaderFlit, PayloadFlit, TailFlit, None]:
        """Returns the flit at the front/left of the queue without removing it."""
        flit = self.queue[0]
        if isinstance(flit, EmptyFlit):
            return None
        else:
            return flit

    def remove(self) -> Union[HeaderFlit, PayloadFlit, TailFlit, None]:
        """
        Removes flits from the buffer if there are any.
        Returns the flit that was removed.
            if there is no flit, returns None.
            else returns the flit that was removed.
        """
        flit = self.queue.popleft()
        if isinstance(flit, EmptyFlit):
            return None

        return flit

    def can_do_routing(self) -> bool:
        """ Checks if the buffer has the complete packet
        to do routing. """
        if self.is_full():
            if isinstance(self.queue[0], HeaderFlit):
                return True
        return False

    def can_transmit_flits(self) -> bool:
        if self.can_do_routing():
            self._in_transmit_mode = True

        if isinstance(self.queue[0], TailFlit):
            self._in_transmit_mode = False
            return True

        return self._in_transmit_mode

    def fill_with_empty_flits(self) -> None:
        """Fill non occupied spaces with Empty Flits"""
        non_occupied_space = self.size - len(self.queue)
        for _ in range(non_occupied_space):
            self.queue.append(EmptyFlit())

    def is_full(self) -> bool:
        non_empty_flit_count = 0
        for flit in self.queue:
            if not isinstance(flit, EmptyFlit):
                non_empty_flit_count += 1

        return non_empty_flit_count == self.size


    def __str__(self):
        queue_str = [str(item) for item in self.queue]
        return f"{queue_str}"

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

    print(f"\nRemoving from \n{buffer}\nStarting")
    for i in range(4):
        buffer.remove()
        buffer.fill_with_empty_flits()
        print(f"{buffer}")

    """
    Test 2: Adding 2 flits to the buffer and then removing them.
    """
    print( "\n- Test 2: Adding 2 flits to the buffer and then removing them." )
    
    packet = Packet(source_xy=(0, 0), 
                    dest_xy=(1, 1), 
                    source_task_id=0)

    buffer = Buffer(4)

    print(f"\nAdding to \n{buffer}\nStarting")
    for i in range(2):
        packet_is_transmitted, flit = packet.transmit_flit()
        buffer.add_flit(flit)
        print(f"{buffer}")

    flit = buffer.remove()
    print( f"Buffer after removing 1 flit: {buffer}, removed flit {flit}" )




