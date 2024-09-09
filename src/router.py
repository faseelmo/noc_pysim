from enum   import Enum
from typing import Union

from .buffer import Buffer
from .flit   import BufferLocation, HeaderFlit, PayloadFlit, TailFlit


class Router:
    def __init__(self, pos: tuple, buffer_size: int = 4):
        """ Args; 
            "pos"           : tuple, coordinates of the router  
            "buffer_size"   : int, number of flits that can be stored in a buffer
        """

        self.x = pos[0]
        self.y = pos[1]

        self.ni_input_buffer        = Buffer( buffer_size )
        self.ni_input_buffer        = Buffer( buffer_size )

        self.west_input_buffer      = Buffer( buffer_size )
        self.west_output_buffer     = Buffer( buffer_size )
  
        self.north_input_buffer     = Buffer( buffer_size )
        self.north_output_buffer    = Buffer( buffer_size )
  
        self.east_input_buffer      = Buffer( buffer_size )
        self.east_output_buffer     = Buffer( buffer_size )
  
        self.south_input_buffer     = Buffer( buffer_size )
        self.south_output_buffer    = Buffer( buffer_size )


    def receive_flits( self, flit: Union[dict, int] ) -> None:

        if isinstance(flit, dict):
            if isinstance(flit, HeaderFlit):
                routing_info    = self._get_routing_information( flit )
                flit["routing"] = routing_info



    def _get_routing_information( self, header_flit: dict) -> list:
        """ Returns the routing information from the flit.
        """
        routing_information = ["Routing Information here"]
        return routing_information

    def __repr__(self):
        return f"R({self.x}, {self.y})"


if __name__ == "__main__":
    
    from .packet import Packet

    router = Router( pos = (0, 0) )
    print( f"Router initialized: { router }")

    packet = Packet( source_xy      = (0, 0), 
                     dest_xy        = (1, 1), 
                     source_task_id = 0 )

    print(f"Packet initialized: {packet}")

    max_sim_cycle = 100

    for i in range( max_sim_cycle ): 

        if i == 0:
            packet_is_transmitted = False # Think about moving this to the packet class

        if not packet_is_transmitted:
            packet_is_transmitted, flit = packet.transmit_flit() 
            router.receive_flits( flit )
            print( f"current flits is {flit}" )

    print( f"Packet transmitted: {packet}" )


    

