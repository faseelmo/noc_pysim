from typing     import Union

from .buffer    import Buffer
from .flit      import HeaderFlit, PayloadFlit, TailFlit, NextHop, BufferLocation


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

        self._input_buffers         = []
        self._output_buffers        = []

        self._populate_buffer_lists()



    def process( self, receive_flit_list: list) -> None:
        """ # Process the flits in the input buffer first 
                # forward_output_buffer_flits()
                # forward_input_buffer_flits()
            # Receive New flits 
                # receive_flits() """

        for flit in receive_flit_list:
            self._receive_flits( flit )



    def _forward_input_buffer_flits( self ) -> None:
        """Moving Flits from input buffer to output buffer."""

        # for buffer in self._input_buffers:
        #     print(f"Buffer status: {buffer}\n")

        # Once Forwarded, remove the routing information from the flit. 

        pass

    def _receive_flits( self, flit: Union[ HeaderFlit, PayloadFlit, TailFlit ]) -> None:
        """
        Receive flits from the PE or other routers.
        Appropriate input buffer is selected based on the routing information. 
        """
        current_routing_info    = flit.get_routing_info()
        assigned_buffer         = current_routing_info.buffer.value

        buffer = getattr( self, f"_{assigned_buffer}_input_buffer" )

        # Here -> Check if the buffer is full

        buffer.add_flit( flit )

        if buffer.can_do_routing(): 

            if not isinstance( flit, TailFlit ):
                raise Exception("Last flit in the packet is not a Tail Flit. Cannot do routing.")

            header_flit_pointer = flit.get_header_pointer()

            next_hop_info    = self._get_routing_information( header_flit_pointer )
            header_flit_pointer.update_routing_info( next_hop_info )

        print(f"{assigned_buffer} buffer status: {buffer}\n")


    def _get_routing_information( self, header_flit: HeaderFlit) -> NextHop:
        """ Returns the routing information from the flit."""
        
        dest_x, dest_y = header_flit.get_destination()

        if dest_x > self._x:    # Destination on east
            next_hop_x  = self._x + 1
            next_buffer = BufferLocation.EAST
            return NextHop( x = next_hop_x, y = self._y, buffer = next_buffer )
            
        elif dest_x < self._x:  # Destination on west
            next_hop_x = self._x - 1
            next_buffer = BufferLocation.WEST
            return NextHop( x = next_hop_x, y = self._y, buffer = next_buffer )

        else:                   # Destination on the same x-axis
            next_hop_x = self._x

        if dest_y > self._y:    # Destination on north
            next_hop_y = self._y + 1
            next_buffer = BufferLocation.NORTH
            return NextHop( x = next_hop_x, y = next_hop_y, buffer = next_buffer )

        elif dest_y < self._y:  # Destination on south
            next_hop_y = self._y - 1
            next_buffer = BufferLocation.SOUTH
            return NextHop( x = next_hop_x, y = next_hop_y, buffer = "south" )

        else:                   # Destination on the same y-axis
            return NextHop( x = next_hop_x, y = next_hop_y, buffer = "local" )


    def _populate_buffer_lists( self ) -> None:
        """Populate the input and output buffers."""
        attributes = vars( self )

        for attr_name, attr_value in attributes.items():

            if "input_buffer" in attr_name:
                self._input_buffers.append( attr_value )

            elif "output_buffer" in attr_name:
                self._output_buffers.append( attr_value )


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

    run_loop    = True
    count       = 0 
    while run_loop:  

        print(f" - Cycle [{count + 1}]")
        count += 1

        packet_is_transmitted, flit = packet.transmit_flit() 
        
        # Idea of flit list is that, when multiple flits from different routers are received,
        # they can be processed in a single cycle.
        flit_list = [flit]
        router.process( flit_list )

        if packet_is_transmitted: 
            run_loop = False


    print( f"Packet transmitted: {packet}" )


    

