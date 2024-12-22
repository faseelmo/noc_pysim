import uuid

from typing         import Union
from collections    import deque

from .flit import HeaderFlit, PayloadFlit, TailFlit, EmptyFlit
from .packet import Packet

class Buffer:
    def __init__(self, size: int, name: str = "Buffer"):
        self.size               = size
        self.queue              = deque(maxlen=size)
        self._name              = name

        self._acceptable_flit_uids = deque(maxlen=2)
        
        self.fill_emtpy_slots()

    def get_name(self) -> str:
        return self._name

    def clear(self) -> None:
        self.queue.clear()
        self._acceptable_flit_uids.clear()
        self.fill_emtpy_slots()

    def add_flit(self, flit: Union[HeaderFlit, PayloadFlit, TailFlit]) -> bool:
        """
        Adds flit to the buffer if it is not full.
        Full is defined as having all non-empty flits.
        """
        if self.is_full(): 
            raise Exception( "Cannot add flit to full buffer." )

        else: 
            if self._is_flit_registered( flit ):
                self.queue.append( flit )
                return True
            else: 

                if self._can_accept_new_packet(): 
                    self._register_flit_uid( flit.get_uid() )
                    self.queue.append( flit )
                    return True 

                else: 
                    raise Exception("Cannot accept new packet and UUID not in acceptable list")

    def can_accept_flit(self, flit: Union[HeaderFlit, PayloadFlit, TailFlit]) -> bool:
        # To do: Call this function in add_flit and remove the if condition from add_flit.
        # I dont wanna do it now, because test conditions will have to be adjusted accordingly. urgh. 
        if self.is_full():
            return False

        if not self._is_flit_registered(flit) and not self._can_accept_new_packet():
            return False

        return True


    def can_transmit_flit(self) -> bool:
        top_flit = self.peek()
        if isinstance(top_flit, HeaderFlit):
            if self.is_full():
                if isinstance(self.queue[-1], TailFlit):
                    return True
                else: 
                    raise Exception("Cannot transmit packet. Tail Flit not in buffer.")
            else: 
                return False
        elif top_flit is None:
            return False
        else: 
            return True


        
    def _register_flit_uid(self, flit_uid: uuid.UUID) -> None:
        header_count    = 0
        tail_count      = 0
        payload_count   = 0

        for flits in self.queue:

            if isinstance(flits, HeaderFlit):
                header_count += 1
            elif isinstance(flits, TailFlit):
                tail_count += 1
            elif isinstance(flits, PayloadFlit):
                payload_count += 1

        assert header_count  <= 1,             f"Invalid Header Count {header_count} in Buffer"
        assert tail_count    <= 1,             f"Invalid Tail Count {tail_count} in Buffer"
        assert payload_count <= self.size - 2, f"Invalid Payload Count {payload_count} in Buffer"

        self._acceptable_flit_uids.append(flit_uid)


    def _can_accept_new_packet(self) -> bool:    
        """
        Conditions to accept a new packet:  
        1. If the acceptable flit uids are empty. 
            (Initial condition)  

        2. If acceptable flit has 1 uid, 
            then the buffer should not have any empty flits.  
            (Condition where the buffer is filled with 1 packet)  

        3. If the buffer has a tail flit, and it has empty flits.
            (Condition where the buffer has the end section of a packet and empty flits)  

        To add to buffer that already has 2 uuid in the acceptable list,  
        top uid should be popped when the tail of that packet is not in the buffer anymore. 
        """
        empty_count     = 0
        has_tail        = False

        if len(self._acceptable_flit_uids) == 2:
            return False

        for flits in self.queue:
            if isinstance(flits, EmptyFlit):
                empty_count += 1

            if isinstance(flits, TailFlit):
                has_tail = True

        if len(self._acceptable_flit_uids) == 0:
            return True

        elif len(self._acceptable_flit_uids) == 1:
            if empty_count == 0 and has_tail:
                return True

        if empty_count > 0 and has_tail:
            return True

        return False

    def _is_flit_registered(self, flit: Union[HeaderFlit, PayloadFlit, TailFlit]) -> bool:
        """If the flit uid is in the acceptable list, return True"""
        flit_uid = flit.get_uid()
        for acceptable_flit_uid in self._acceptable_flit_uids:
            if acceptable_flit_uid == flit_uid:
                return True

        return False


    def fill_with_packet(self, packet: Packet) -> None:
        
        if not self._can_accept_new_packet():
            Exception("Cannot accept new packet. Buffer is not in a state to accept a full packet.")

        while True: 
            packet_is_transmitted, flit = packet.pop_flit()
            self.add_flit(flit)
            if packet_is_transmitted:
                break

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

        if isinstance(flit, TailFlit):
            self._acceptable_flit_uids.popleft()

        return flit

    def can_do_routing(self) -> bool:
        """ Checks if the buffer has the complete packet
        to do routing. """
        if self.is_full():
            if isinstance(self.queue[0], HeaderFlit):
                return True
        return False

    def fill_emtpy_slots(self, n: int = 0) -> None:
        """
        Fill non occupied spaces with Empty Flits
        Arg n: Number of spaces that should remain unfilled with Empty Flits.
        """
        non_occupied_space = self.size - len(self.queue) - n 
        for _ in range(non_occupied_space):
            self.queue.append(EmptyFlit())

    def manager(self) -> None:

        empty_flit_count = 0
        for flit in self.queue:
            if isinstance(flit, EmptyFlit):
                empty_flit_count += 1
        
        # If the buffer has all empty flits and less than the buffer size
        # fill it to the brim with empty flits.
        if len(self.queue) < self.size and len(self.queue) == empty_flit_count:
            self.fill_emtpy_slots()

        # If there are no empty flits and the buffer is not full (one slot remaining),
        # fill the buffer with empty flits, leaving one slot empty.
        if empty_flit_count == 0:
            # if len(self.queue) < (self.size - 1): 
            # if len(self.queue) < (self.size - 1): 
                self.fill_emtpy_slots(1)


    def is_full(self) -> bool:
        """
        Returns True if the buffer is full  
        Full is defined as having all non - EmptyFlit.
        """
        if len(self.queue) == 0:
            return False

        if isinstance(self.queue[0], EmptyFlit) :
            return False

        elif len(self.queue) < self.size:
            return False    

        return True

    def is_empty(self) -> bool:
        """
        Returns True if the buffer is all EmptyFlit.
        """
        empty_flit_count = 0
        for flit in self.queue:
            if isinstance(flit, EmptyFlit):
                empty_flit_count += 1
        return empty_flit_count == self.size

    def empty(self) -> None:
        """Empty the buffer"""
        if isinstance(self.queue[0], HeaderFlit) and isinstance(self.queue[-1], TailFlit):
            for _ in range(len(self.queue)):
                self.queue.append(EmptyFlit())
            self._acceptable_flit_uids.clear()

        else: 
            raise Exception("Cannot Empty Buffer. Buffer does not have a complete packet.")


    def __str__(self):
        queue_str = [str(item) for item in self.queue]
        return f"{self._name} {queue_str}"

if __name__ == "__main__":

    from .packet import Packet

    buffer = Buffer(4)
    print(f"\n{buffer}")

    """
    Test 1: Adding 4 flits to the buffer and then removing them.
    """
    print( "\n- Test 1: Adding 4 flits to the buffer and then removing them." )
    packet = Packet(source_xy       = (0, 0), 
                    dest_id         = 1, 
                    source_task_id  = 0)

    packet_is_transmitted = False
    while not packet_is_transmitted:
        packet_is_transmitted, flit = packet.pop_flit()
        buffer.add_flit(flit)
        print(f"{buffer}")

    print(f"\nRemoving from \n{buffer}\nStarting")
    for i in range(4):
        buffer.remove()
        buffer.fill_emtpy_slots()
        print(f"{buffer}")

    """
    Test 2: Adding 2 flits to the buffer and then removing them.
    """
    print( "\n- Test 2: Adding 2 flits to the buffer and then removing them." )
    
    packet = Packet(source_xy       = (0, 0), 
                    dest_id         = 1, 
                    source_task_id  = 0)

    buffer = Buffer(4)

    print(f"\nAdding to \n{buffer}\nStarting")
    for i in range(2):
        packet_is_transmitted, flit = packet.pop_flit()
        buffer.add_flit(flit)
        print(f"{buffer}")

    flit = buffer.remove()
    print( f"Buffer after removing 1 flit: {buffer}, removed flit {flit}" )




