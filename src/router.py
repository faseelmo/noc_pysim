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


    def process( self, receive_flit_list: list, router_lookup: dict) -> list[ Union[ HeaderFlit, PayloadFlit, TailFlit ] ]:
        """ - Process the flits in the input buffer first 
                - forward_output_buffer_flits()
                - forward_input_buffer_flits()
            - Receive New flits 
                - receive_flits() """

        new_flit_list = self._forward_output_buffer_flits( router_lookup )

        self._forward_input_buffer_flits()

        for flit in receive_flit_list:
            self._receive_flit( flit )

        return new_flit_list

    def is_local_input_buffer_full(self) -> bool:
        return self._local_input_buffer.is_full()

    def add_flit_to_local_input_buffer(self, flit: Union[HeaderFlit, PayloadFlit, TailFlit]) -> None:
        self._local_input_buffer.add_flit(flit)

    def _forward_output_buffer_flits( self, router_lookup: dict ) -> list[ Union[ HeaderFlit, PayloadFlit, TailFlit ] ]:
        """
        Check if the output has any flits to be forwarded to the next router.
        If there are, check if the next router has space in the input buffer.
        If it does, remove the flit from the output buffer and return it. 
        """

        flit_list = []

        for buffer in self._output_buffers:
            top_flit = buffer.peek()

            if top_flit is None:
                continue

            next_hop_x      = top_flit.get_routing_info().x
            next_hop_y      = top_flit.get_routing_info().y
            next_hop_loc    = (next_hop_x, next_hop_y)
            next_hop_buffer = top_flit.get_routing_info().output_buffer

            next_router = router_lookup.get( next_hop_loc )
            
            if next_router == self:
                # Condition when the flit has reached the destination router
                # think about how to move the flit to the PE
                continue

            next_router_input_buffer = next_router._get_buffer( direction = next_hop_buffer, is_input = True )

            if not next_router_input_buffer.is_full():
                flit = buffer.remove()
                flit_list.append( flit )

            buffer.fill_emtpy_slots()  

        return flit_list 

    def _forward_input_buffer_flits( self ) -> None:
        """
        Forwarding Flits from input buffer to the output buffer of the same router.  
        Functionality: Iterates through the input buffers and pops a flit from the buffer.
        if the flit is not None, it is forwarded to the next buffer.
        for flits from the PE, routing information is updated.
        """

        for buffer in self._input_buffers:
            top_flit   = buffer.peek()

            if top_flit is None:
                continue

            next_hop_location   = top_flit.get_routing_info().output_buffer

            # Check if routing information is available for packets copied from the PE
            if next_hop_location == BufferLocation.UNASSIGNED:
                self._do_routing_for_flits_from_pe( buffer )
                next_hop_location   = top_flit.get_routing_info().output_buffer

            next_buffer         = self._get_buffer( direction = next_hop_location, is_input = False )

            if not next_buffer.is_full():
                flit = buffer.remove()
                next_buffer.add_flit( flit )    
                print(f"[{self}](Input Buffer -> Output Buffer Forward)")
                print(f"\t-> {next_hop_location.value} output: {next_buffer}")

            buffer.fill_emtpy_slots()


    def _do_routing_for_flits_from_pe( self, buffer: Buffer ) -> None:
        print(f"Doing routing for flits from PE")
        tail_flit = buffer.queue[-1]
        self._update_routing( tail_flit )

    def _receive_flit( self, flit: Union[ HeaderFlit, PayloadFlit, TailFlit ]) -> None:
        """
        Receive flits from the PE or other routers.
        Appropriate input buffer is selected based on the routing information. 
        """
        print(f"[{self}](Receive Flits)")

        current_routing_info    = flit.get_routing_info()
        buffer_location         = current_routing_info.next_input_buffer

        input_buffer            = self._get_buffer( direction = buffer_location, is_input = True )  

        assert not input_buffer.is_full(), f"Buffer {buffer_location.value} is full. Cannot receive flit."

        input_buffer.add_flit( flit )
        print(f"\t->{buffer_location.value} input: {input_buffer}")

        if input_buffer.can_do_routing(): 
            self._update_routing( flit )


    def _update_routing( self, flit: TailFlit) -> None: 
        """
        Updates the routing information of the flit based on the header flit destination.
        Calls _xy_routing function
        """

        if not isinstance( flit, TailFlit ):
            raise Exception("Error in Packet. Last flit in the packet is not a TailFlit. Cannot do routing.")

        header_flit_pointer = flit.get_header_pointer() # Flit here is a TailFlit

        next_hop_info       = self._xy_routing( header_flit_pointer )

        header_flit_pointer.update_routing_info( next_hop_info )
        print(f"\t\t\tRouting Information Updated: {next_hop_info}")

    def _get_buffer(self, direction:BufferLocation, is_input:bool) -> Buffer:
        """Returns the buffer based on the direction(str) and input/output flag (bool)."""
        direction_str = direction.value
        attributes = vars( self )
        for attr_name, attr_value in attributes.items():
            if isinstance( attr_value, Buffer ):
                if direction_str in attr_name:
                    if is_input and "input" in attr_name:
                        return attr_value
                    elif not is_input and "output" in attr_name:
                        return attr_value

    def _xy_routing( self, header_flit: HeaderFlit) -> NextHop:
        """ 
        Returns the routing information from the flit.
        Computes the x and y coordinates of the next hop based on the destination.
        Also computes which buffer the flit should be forwarded to.
        """

        dest_x, dest_y = header_flit.get_destination()
        
        # For X-axis
        if dest_x > self._x:    # Destination on east
            next_hop_x  = self._x + 1
            return NextHop( 
                x                   = next_hop_x, 
                y                   = self._y, 
                output_buffer       = BufferLocation.EAST, 
                next_input_buffer   = BufferLocation.WEST )
            
        elif dest_x < self._x:  # Destination on west
            next_hop_x  = self._x - 1
            return NextHop( 
                x                   = next_hop_x, 
                y                   = self._y, 
                output_buffer       = BufferLocation.WEST, 
                next_input_buffer   = BufferLocation.EAST )

        else:                   # Destination on the same x-axis
            next_hop_x = self._x

        # For Y-axis
        if dest_y > self._y:    # Destination on north
            next_hop_y  = self._y + 1
            return NextHop( 
                x                   = next_hop_x, 
                y                   = next_hop_y, 
                output_buffer       = BufferLocation.NORTH, 
                next_input_buffer   = BufferLocation.SOUTH )

        elif dest_y < self._y:  # Destination on south
            next_hop_y  = self._y - 1
            return NextHop( 
                x                   = next_hop_x, 
                y                   = next_hop_y, 
                output_buffer       = BufferLocation.SOUTH, 
                next_input_buffer   = BufferLocation.NORTH )

        else:                   # Destination on the same y-axis
            return NextHop( 
                x                   = next_hop_x, 
                y                   = self._y, 
                output_buffer       = BufferLocation.LOCAL, 
                next_input_buffer   = BufferLocation.UNASSIGNED ) # Going to the PE


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
    from .processing_element import ProcessingElement, TaskInfo

    """
    Condition 1 : 
        - Packet comes from a IP Core, without any routing information.  
        - Look up the destination and compute the routing information (i.e next hop)  
        - And then move it to appropriate buffer.   

        To dos: 
        [x] Implement copying the packet to the buffer.
        [x] Implement the routing algorithm.
        [x] Implement the moving to the next buffer.
        [x] Implement looking at the output of all the routers and 
            appending the flits to a list which can be processed in the 
            next router at the next cycle. 
        [x] Use the flit from the list to process at the next router. 
        [x] Improve logging. 

    """

    router_00 = Router( pos = (0, 0) )
    router_10 = Router( pos = (1, 0) )
    router_11 = Router( pos = (1, 1) )

    router_lookup = { (0, 0): router_00, (1, 0): router_10, (1, 1): router_11 }

    task = TaskInfo(
            task_id                     = 0, 
            processing_cycles           = 4, 
            expected_generated_packets  = 1, 
            require_list                = [], 
            is_transmit_task            = True, 
            transmit_dest_xy            = (1, 1)
    )

    pe_00 = ProcessingElement( xy = (0, 0), computing_list = [ task ], debug_mode=True, router_lookup = router_lookup )

    pe_lookup = { (0, 0): pe_00 }

    flit_list = []

    for i in range(35): 
        print(f"\n> {i}")
        pe_00.process(None)

        for router in router_lookup.values():
            flit_list = router.process( flit_list, router_lookup )  


