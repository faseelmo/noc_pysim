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
        self._local_output_buffer   = Buffer( buffer_size )

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
        """ - Process the flits in the input buffer first 
                - forward_output_buffer_flits()
                - forward_input_buffer_flits()
            - Receive New flits 
                - receive_flits() """
        self._forward_input_buffer_flits()

        for flit in receive_flit_list:
            self._receive_flits( flit )


        for buffer in self._input_buffers:
            buffer.fill_with_empty_flits()

        for buffer in self._output_buffers:
            buffer.fill_with_empty_flits()



    def _forward_input_buffer_flits( self ) -> None:
        """Moving Flits from input buffer to output buffer."""

        for buffer in self._input_buffers:
            flit        = buffer.remove() 

            if flit is None:
                continue

            next_hop_location   = flit.get_routing_info().buffer.value
            next_buffer         = self._get_buffer( direction = next_hop_location, is_input = False )
            next_buffer.add_flit( flit )    
            print(f"{next_hop_location} buffer status: {next_buffer}\n")

        # Once Forwarded, remove the routing information from the flit. 

        pass

    def _receive_flits( self, flit: Union[ HeaderFlit, PayloadFlit, TailFlit ]) -> None:
        """
        Receive flits from the PE or other routers.
        Appropriate input buffer is selected based on the routing information. 
        """
        current_routing_info    = flit.get_routing_info()
        buffer_location         = current_routing_info.buffer.value

        buffer_in_curr_router   = getattr( self, f"_{buffer_location}_input_buffer" )

        # Here -> Check if the buffer is full. Do it from the External side.

        buffer_in_curr_router.add_flit( flit )

        if buffer_in_curr_router.can_do_routing(): 

            if not isinstance( flit, TailFlit ):
                raise Exception("Error in Packet. Last flit in the packet is not a TailFlit. Cannot do routing.")

            # Flit here is a TailFlit
            header_flit_pointer = flit.get_header_pointer()

            next_hop_info    = self._get_routing_information( header_flit_pointer )
            header_flit_pointer.update_routing_info( next_hop_info )

        print(f"{buffer_location} buffer status: {buffer_in_curr_router}\n")

    def _get_buffer(self, direction:str, is_input:bool) -> Buffer:
        """Returns the buffer based on the direction(str) and input/output flag (bool)."""
        attributes = vars( self )
        for attr_name, attr_value in attributes.items():
            if isinstance( attr_value, Buffer ):
                if direction in attr_name:
                    if is_input and "input" in attr_name:
                        return attr_value
                    elif not is_input and "output" in attr_name:
                        return attr_value

    def _get_routing_information( self, header_flit: HeaderFlit) -> NextHop:
        """ Returns the routing information from the flit."""
        print(f"Calculating routing information")

        dest_x, dest_y = header_flit.get_destination()

        if dest_x > self._x:    # Destination on east
            next_hop_x  = self._x + 1
            next_buffer = BufferLocation.EAST
            return NextHop( x = next_hop_x, y = self._y, buffer = next_buffer )
            
        elif dest_x < self._x:  # Destination on west
            next_hop_x  = self._x - 1
            next_buffer = BufferLocation.WEST
            return NextHop( x = next_hop_x, y = self._y, buffer = next_buffer )

        else:                   # Destination on the same x-axis
            next_hop_x = self._x

        if dest_y > self._y:    # Destination on north
            next_hop_y  = self._y + 1
            next_buffer = BufferLocation.NORTH
            return NextHop( x = next_hop_x, y = next_hop_y, buffer = next_buffer )

        elif dest_y < self._y:  # Destination on south
            next_hop_y  = self._y - 1
            next_buffer = BufferLocation.SOUTH
            return NextHop( x = next_hop_x, y = next_hop_y, buffer = "south" )

        else:                   # Destination on the same y-axis
            return NextHop( x = next_hop_x, y = next_hop_y, buffer = "local" )


    def _populate_buffer_lists( self ) -> None:
        """Copies each buffer to the respective list (input or output). """
        attributes = vars( self )

        for attr_name, attr_value in attributes.items():

            if "input_buffer" in attr_name:
                if isinstance( attr_value, Buffer ):
                    self._input_buffers.append( attr_value )

            elif "output_buffer" in attr_name:
                if isinstance( attr_value, Buffer ):
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
        [x] Implement the routing algorithm.
        [x] Implement the moving to the next buffer.

    """

    router = Router( pos = (0, 0) )
    print( f"\nRouter initialized: { router }" )

    packet = Packet( source_xy      = (0, 0), 
                     dest_xy        = (1, 1), 
                     source_task_id = 0     )
    print( f"\nPacket initialized: {packet}\n" )

    max_sim_cycle           = 10
    cycle                   = 0 

    # Idea of flit list is that, when multiple flits from different routers are received,
    # they can be processed in a single cycle.
    flit_list               = []
    stop_sending_flits      = False

    while cycle < max_sim_cycle:  

        print(f" - Cycle [{cycle + 1}]")
        cycle += 1

        packet_is_transmitted, flit = packet.transmit_flit() 
        if not stop_sending_flits:
            flit_list = [flit]

        router.process( flit_list )

        if packet_is_transmitted: 
            stop_sending_flits = True
            flit_list = []




    

