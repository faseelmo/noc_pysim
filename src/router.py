from typing     import Union

from .buffer    import Buffer
from .flit      import HeaderFlit, PayloadFlit, TailFlit, NextHop


class Router:
    def __init__(self, pos: tuple, buffer_size: int = 4):
        """ Args; 
            "pos"           : tuple, coordinates of the router  
            "buffer_size"   : int, number of flits that can be stored in a buffer
        """

        self._x = pos[0]
        self._y = pos[1]

        self._local_input_buffer    = Buffer( buffer_size )
        self._ni_input_buffer       = Buffer( buffer_size )

        self._west_input_buffer     = Buffer( buffer_size )
        self._west_output_buffer    = Buffer( buffer_size )

        self._north_input_buffer    = Buffer( buffer_size )
        self._north_output_buffer   = Buffer( buffer_size )

        self._east_input_buffer     = Buffer( buffer_size )
        self._east_output_buffer    = Buffer( buffer_size )

        self._south_input_buffer    = Buffer( buffer_size )
        self._south_output_buffer   = Buffer( buffer_size )


    def process_flits( self ) -> None:
        # Process the flits in the input buffer first 
            # forward_flits()
        # Receive New flits 
            # receive_flits()
        pass

    def forward_flits( self ) -> None:
        """Moving Flits from input buffer to output buffer."""

        pass

    def receive_flits( self, flit: Union[ HeaderFlit, PayloadFlit, TailFlit ], buffer_name: str ) -> None:

        buffer = getattr( self, f"_{buffer_name}_input_buffer" )

        # Here -> Check if the buffer is full

        if isinstance(flit, HeaderFlit):
            next_hop_info    = self._get_routing_information( flit )
            flit.update_routing_info( next_hop_info )

        buffer.add_flit( flit )
        print(f"Buffer status: {buffer}\n")


    def _get_routing_information( self, header_flit: dict) -> NextHop:
        """ Returns the routing information from the flit.
        """
        
        dest_x, dest_y = header_flit.get_destination()

        if dest_x > self._x:    # Destination on east
            next_hop_x  = self._x + 1
            next_buffer = "east"
            return NextHop( x = next_hop_x, y = self._y, buffer = next_buffer )
            
        elif dest_x < self._x:  # Destination on west
            next_hop_x = self._x - 1
            next_buffer = "west"
            return NextHop( x = next_hop_x, y = self._y, buffer = next_buffer )

        else:                   # Destination on the same x-axis
            next_hop_x = self._x

        if dest_y > self._y:
            next_hop_y = self._y + 1
            next_buffer = "north"
            return NextHop( x = next_hop_x, y = next_hop_y, buffer = next_buffer )

        elif dest_y < self._y:
            next_hop_y = self._y - 1
            return NextHop( x = next_hop_x, y = next_hop_y, buffer = "south" )

        else: 
            return NextHop( x = next_hop_x, y = next_hop_y, buffer = "local" )


    def __repr__(self):
        return f"R({self._x}, {self._y})"


if __name__ == "__main__":

    from .packet import Packet

    max_sim_cycle = 100

    """
    Condition 1 : 
        - Packet comes from a IP Core, without any routing information.  
        - Look up the destination and compute the routing information (i.e next hop)  
        - And then move it to appropriate buffer.   

        To dos: 
        [x] Implement copying the packet to the buffer.
        [ ] Implement the routing algorithm.

    """

    router = Router( pos = (0, 0) )
    print( f"\nRouter initialized: { router }")

    packet = Packet( source_xy      = (0, 0), 
                     dest_xy        = (1, 1), 
                     source_task_id = 0 )
    print(f"\nPacket initialized: {packet}\n")

    for i in range( max_sim_cycle ): 

        print(f"Cycle: {i}")
        if i == 0:
            packet_is_transmitted = False # Think about moving this to the packet class

        if not packet_is_transmitted:
            packet_is_transmitted, flit = packet.transmit_flit() 
            router.receive_flits( flit, "local" )
            print( f"current flits is {flit}" )

        else: break

    print( f"Packet transmitted: {packet}" )


    

