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

        self._forward_output_buffer_flits()

        self._forward_input_buffer_flits()

        for flit in receive_flit_list:
            self._receive_flit( flit )


    def _forward_output_buffer_flits( self ) -> None:
        """
        Check if the output has any flits to be forwarded to the next router.
        If there are, check if the next router has space in the input buffer.
        If it does, remove the flit from the output buffer and return it. 
        """
        print(f"[{self}](Output Buffer -> Router Forward)")

        for buffer in self._output_buffers:
            buffer.fill_with_empty_flits()  
       


    def _forward_input_buffer_flits( self ) -> None:
        """
        Forwarding Flits from input buffer to the output buffer of the same router.  
        Functionality: Iterates through the input buffers and pops a flit from the buffer.
        if the flit is not None, it is forwarded to the next buffer.
        """
        print(f"[{self}](Input Buffer -> Output Buffer Forward)")

        for buffer in self._input_buffers:
            top_flit   = buffer.peek()

            if top_flit is None:
                continue

            next_hop_location   = top_flit.get_routing_info().buffer.value
            next_buffer         = self._get_buffer( direction = next_hop_location, is_input = False )

            if not next_buffer.is_full():
                flit = buffer.remove()
                next_buffer.add_flit( flit )    
                print(f"\t\t-> {next_hop_location} buffer: {next_buffer}")

            buffer.fill_with_empty_flits()


    def _receive_flit( self, flit: Union[ HeaderFlit, PayloadFlit, TailFlit ]) -> None:
        """
        Receive flits from the PE or other routers.
        Appropriate input buffer is selected based on the routing information. 
        """
        print(f"[{self}](Receive Flits)")

        current_routing_info    = flit.get_routing_info()
        buffer_location         = current_routing_info.buffer.value

        buffer_in_curr_router   = getattr( self, f"_{buffer_location}_input_buffer" )

        assert not buffer_in_curr_router.is_full(), f"Buffer {buffer_location} is full. Cannot receive flit."

        buffer_in_curr_router.add_flit( flit )
        print(f"\t\t{buffer_location} -> Buffer status: {buffer_in_curr_router}")

        if buffer_in_curr_router.can_do_routing(): 
            self._do_routing( flit )


    def _do_routing( self, flit: TailFlit) -> None: 

        if not isinstance( flit, TailFlit ):
            raise Exception("Error in Packet. Last flit in the packet is not a TailFlit. Cannot do routing.")

        header_flit_pointer = flit.get_header_pointer() # Flit here is a TailFlit

        next_hop_info       = self._get_routing_information( header_flit_pointer )

        header_flit_pointer.update_routing_info( next_hop_info )
        print(f"\t\t\tRouting Information Updated: {next_hop_info}")

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
        """ 
        Returns the routing information from the flit.
        Computes the x and y coordinates of the next hop based on the destination.
        Also computes which buffer the flit should be forwarded to.
        """

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


    def __eq__(self, other):
        """
        Overriding the equality operator to compare the x and y coordinates of the router.
        Argument "other" can be a Router object or a tuple.
        In both cases, the x and y coordinates are compared.
        """
        if isinstance(other, Router):
            return self._x == other._x and self._y == other._y
        elif isinstance(other, tuple):
            return self._x == other[0] and self._y == other[1] 
        return False

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
        [x] Implement looking at the output buffers and moving the flits to the input 
            of the destination router's input buffer. 

    """

    print(f"\nStarting Simulation with {max_sim_cycle} cycles.\n")
    
    
    router_00 = Router( pos = (0, 0) )
    router_10 = Router( pos = (1, 0) )
    router_11 = Router( pos = (1, 1) )

    router_lookup = { (0, 0): router_00, (1, 0): router_10, (1, 1): router_11 }

    for router in router_lookup.values():
        print( f" - Router initialized: { router }" )

    packet = Packet( source_xy      = (0, 0), 
                     dest_xy        = (1, 1), 
                     source_task_id = 0     )
    print( f"\n - Packet initialized: {packet}\n" )

    max_sim_cycle           = 10
    cycle                   = 0 

    # Idea of flit list is that, when multiple flits from different routers are received,
    # they can be processed in a single cycle.
    flit_list               = []
    stop_sending_flits      = False

    while cycle < max_sim_cycle:  

        print(f"\n - Cycle [{cycle + 1}]")
        cycle += 1

        packet_is_transmitted, flit = packet.transmit_flit() 
        dest_router = (flit.get_routing_info().x, flit.get_routing_info().y) 

        if not stop_sending_flits:
            flit_list = [flit]

        for router in router_lookup.values():

            # print(f"\n • Router: {router}")
            if router == (0,0):
                router.process( flit_list )

            # else : 
            #     router.process( [] )
    
            if packet_is_transmitted: 
                stop_sending_flits = True
                flit_list = []




    

