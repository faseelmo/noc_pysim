import uuid
from enum import Enum
from dataclasses    import dataclass   

class BufferLocation(Enum):
    NORTH       = "north"
    WEST        = "west"
    SOUTH       = "south"
    EAST        = "east"
    LOCAL       = "local"
    UNASSIGNED  = "unassigned" 

@dataclass
class NextHop:
    """
    'next_input_buffer' is the buffer of the next router. 
        if the flit is coming from a PE it is set to LOCAL 

    'output_buffer' is the output buffer of the current router 
    which the flit will be sent before it reaches the 'next_input_buffer' 
    of the next router.
    """
    x                   : int 
    y                   : int 
    next_input_buffer   : BufferLocation 
    output_buffer       : BufferLocation = BufferLocation.UNASSIGNED 

    def __str__(self):
        return (
            f"Next Hop -> R({self.x}, {self.y}) " 
            f"next_input_buffer: {self.next_input_buffer.value}, "
            f"output_buffer: {self.output_buffer.value}"
        )


class HeaderFlit: 
    def __init__( self, src_xy: tuple, dest_id: int, packet_uid: uuid.UUID, source_task_id: int ): 
        """
        - When HeaderFlit is created, it is assigned a next_hop attribute.
        - The next_hop attribute include the x,y coordinates of the next hop. 
        - Since all the packets are created at a core attached to the router 
          with the same x,y coordinates, the next_hop attribute is initialized
          with the same x,y coordinates. 
          Also, the buffer attribute is set to Local. 
        """

        self._src_xy            = src_xy
        self._dest_id           = dest_id
        self._packet_uid        = packet_uid
        self._source_task_id    = source_task_id
        self._next_hop          = NextHop( 
                                    x=src_xy[0], 
                                    y=src_xy[1], 
                                    next_input_buffer=BufferLocation.LOCAL )

    def update_routing_info( self, next_hop: NextHop ) -> None:
        self._next_hop = next_hop

    def get_destination( self ) -> tuple:
        """ returns destination (x, y) coordinates"""
        return self._dest_id

    def get_routing_info( self ) -> NextHop: 
        return self._next_hop

    def clear_routing_info( self ) -> None:
        self._next_hop.output_buffer = BufferLocation.UNASSIGNED
        self._next_hop.next_input_buffer = BufferLocation.UNASSIGNED

    def get_uid( self ) -> uuid.UUID:
        return self._packet_uid

    def get_source_task_id( self ) -> int:  
        return self._source_task_id

    def __eq__(self, value):
        if isinstance(value, HeaderFlit):
            return self._packet_uid == value.get_uid()
        return False

    def __str__( self ): 
        return ( f"[Header Flit] (task: {self._source_task_id} -> {self._dest_id})" )

class BaseFlit: 
    def __init__(self, header_flit=None):
        self._header_flit   = header_flit 

    def get_uid(self) -> uuid.UUID:
        return self._header_flit.get_uid()

    def get_source_task_id(self) -> int:
        return self._header_flit.get_source_task_id()

    def get_routing_info(self) -> NextHop:
        return self._header_flit.get_routing_info()

    def __eq__(self, value):
        if type(value) == type(self):
            return self.get_uid() == value.get_uid()
        return False

class PayloadFlit(BaseFlit):
    def __init__(self, payload_index: int, header_flit: HeaderFlit):
        super().__init__( header_flit )
        self._payload_index = payload_index

    def __str__(self):
        return f"[Payload Flit] idx: {self._payload_index}"


class TailFlit(BaseFlit):
    def __init__(self, header_flit: HeaderFlit):
        super().__init__( header_flit )

    def get_header_pointer(self) -> HeaderFlit:
        """Get the pointer to the associated header. 
        Used for updating the routing information in the header flit."""
        return self._header_flit

    def __str__(self):
        # return f"[Tail Flit] UUID: {self._packet_uid}"
        return f"[Tail Flit] (task: {self.get_source_task_id()} -> {self.get_header_pointer().get_destination()})"


class EmptyFlit:
    def __str__(self):
        return f"[Empty Flit]"
        
