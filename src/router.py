from typing     import Union

from .buffer    import Buffer
from .flit      import HeaderFlit, PayloadFlit, TailFlit, NextHop, BufferLocation

class Router:
    def __init__( self, pos: tuple, buffer_size: int = 4, debug_mode: bool = False ):
        """ Args; 
            "pos"           : tuple, coordinates of the router  
            "buffer_size"   : int, number of flits that can be stored in a buffer
        """
        self._x = pos[0]
        self._y = pos[1]

        self._debug_mode = debug_mode

        self._local_input_buffer    = Buffer( buffer_size, name="local_input"  )
        self._local_output_buffer   = Buffer( buffer_size, name="local_output" )

        self._west_input_buffer     = Buffer( buffer_size, name="west_input"  )
        self._west_output_buffer    = Buffer( buffer_size, name="west_output" )

        self._north_input_buffer    = Buffer( buffer_size, name="north_input"  )
        self._north_output_buffer   = Buffer( buffer_size, name="north_output" )

        self._east_input_buffer     = Buffer( buffer_size, name="east_input"  )
        self._east_output_buffer    = Buffer( buffer_size, name="east_output" )

        self._south_input_buffer    = Buffer( buffer_size, name="south_input"  ) 
        self._south_output_buffer   = Buffer( buffer_size, name="south_output" )

        self._input_buffers         = []
        self._output_buffers        = []

        self._mapping_list          = []

        self._populate_buffer_lists()

    def clear(self) -> None:
        """Clears the buffers of the router."""
        for buffer in self._input_buffers:
            buffer.clear()

        for buffer in self._output_buffers:
            buffer.clear()

        self._mapping_list.clear()

    def process( self ) -> None:
        """ - Process the flits in the input buffer first 
                - forward_input_buffer_flits()
            - Receive New flits 
                - receive_flits() """

        self._forward_input_buffer_flits()
        self.management()


    def _forward_output_and_process( self, router_lookup, pe_lookup) -> None:
        """
        BACKWARD COMPATIBILITY (ONLY FOR PYTEST)
        Forward flits from the output buffer to the next router.
        Process the flits in the input buffer.
        Receive new flits.
        """
        self.forward_output_buffer_flits( router_lookup, pe_lookup )
        self.process( )


    def set_mapping_list(self, mapping_list: list) -> None:
        """
        Needs mapping list to compute the routing of packets based on destination 
        task id. 
        """
        self._mapping_list = mapping_list

    def _filter_required_flits( self, flit_list: list[Union[HeaderFlit, PayloadFlit, TailFlit]] ) -> list[Union[HeaderFlit, PayloadFlit, TailFlit]]:
        """Filter the flits that are required by the router."""
        filtered_flits = []
        for flit in flit_list:
            next_hop = flit.get_routing_info()
            if (next_hop.x, next_hop.y) == (self._x, self._y):
                filtered_flits.append(flit)
        return filtered_flits

    def is_local_input_buffer_full(self) -> bool:
        return self._local_input_buffer.is_full()

    def add_flit_to_local_input_buffer(self, flit: Union[HeaderFlit, PayloadFlit, TailFlit]) -> None:
        self._local_input_buffer.add_flit(flit)

    def forward_output_buffer_flits( self, router_lookup: dict, pe_lookup: dict ) -> None:
        """
        Check if the output has any flits to be forwarded to the next router.
        If there are, check if the next router has space in the input buffer.
        If it does, remove the flit from the output buffer and return it. 
        """

        for buffer in self._output_buffers:
            top_flit = buffer.peek()

            if top_flit is None:
                continue

            next_hop_x      = top_flit.get_routing_info().x
            next_hop_y      = top_flit.get_routing_info().y
            next_hop_loc    = (next_hop_x, next_hop_y)
            next_hop_buffer = top_flit.get_routing_info().next_input_buffer

            next_router = router_lookup.get( next_hop_loc )

            if next_router == self:
                # Condition when the flit has reached the destination router
                pe = pe_lookup.get( next_hop_loc )

                if not pe.is_input_buffer_full():
                    self._debug_print( f"Forwading: "+ f"{buffer}".split()[0] +" -> PE" )
                    flit = buffer.remove()
                    pe.receive_flits( flit )
                    # buffer.fill_emtpy_slots()

                self._debug_print( f"Local output: {buffer}" )
                continue

            next_router_input_buffer = next_router._get_buffer( direction = next_hop_buffer, is_input = True )

            if not next_router_input_buffer.is_full():
                # Checking if new Header can be registered in the input buffer of 
                # the next router.
                if isinstance(top_flit, HeaderFlit):
                    if not next_router_input_buffer._can_accept_new_packet(): 
                        continue

                self._debug_print( 
                    f"Forwarding flit \"{top_flit}\" "  +
                    f"{buffer}".split()[0] + f"-> " +
                    f"{next_router_input_buffer}".split()[0] + 
                    f" {next_router}" )

                flit = buffer.remove()
                next_router._receive_flit( flit )


    def _forward_input_buffer_flits( self ) -> None:
        """
        Forwarding Flits from input buffer to the output buffer of the same router.  
        Functionality: Iterates through the input buffers and pops a flit from the buffer.
        if the flit is not None, it is forwarded to the next buffer.
        for flits from the PE, routing information is updated.
        """

        for buffer in self._input_buffers:

            top_flit   = buffer.peek()

            if not buffer.can_transmit_flit(): 
                continue    


            next_hop_location   = top_flit.get_routing_info().output_buffer

            # Check if routing information is available for packets copied from the PE
            if next_hop_location == BufferLocation.UNASSIGNED and isinstance(top_flit, HeaderFlit):
            # if next_hop_location == BufferLocation.UNASSIGNED:
                if not buffer.can_do_routing():
                    continue

                self._compute_routing( buffer )
                next_hop_location   = top_flit.get_routing_info().output_buffer

                if buffer.get_name() == "local_input": 
                    continue 

            next_buffer = self._get_buffer( direction = next_hop_location, is_input = False )

            if next_buffer.can_accept_flit(top_flit):

                self._debug_print( 
                    f"Forwading flit \"{top_flit}\" from " + f"{buffer}".split()[0] + f"-> {next_hop_location.value}_output")

                flit = buffer.remove()
                next_buffer.add_flit( flit )    

                self._debug_print(f"\t-> {next_buffer}", with_tag=False)


    def management( self ) -> None:
        
        for buffer in self._input_buffers:
            buffer.manager()

        for buffer in self._output_buffers:
            buffer.manager()

    def _compute_routing( self, buffer: Buffer ) -> None:
        tail_flit = buffer.queue[-1]
        self._update_routing( tail_flit, f"{buffer}".split()[0] )

    def _receive_flit( self, flit: Union[ HeaderFlit, PayloadFlit, TailFlit ]) -> None:
        """
        Receive flits from the PE or other routers.
        Appropriate input buffer is selected based on the routing information. 
        """

        current_routing_info    = flit.get_routing_info()
        buffer_location         = current_routing_info.next_input_buffer

        input_buffer            = self._get_buffer( direction = buffer_location, is_input = True )  

        assert not input_buffer.is_full(), f"{self} {buffer_location.value} buffer is full. Cannot receive flit."

        input_buffer_name = f"{input_buffer}".split()[0]
        self._debug_print( f"Received flit to {input_buffer_name}" )

        input_buffer.add_flit( flit )
        self._debug_print(f"\t-> {input_buffer}", with_tag=False)

        if isinstance(flit, TailFlit):
            self._debug_print( f"Clearing routing info of the packet in {input_buffer_name}" )
            flit.get_header_pointer().clear_routing_info()


    def _update_routing( self, flit: TailFlit, buffer_name: str) -> None: 
        """
        Updates the routing information of the flit based on the header flit destination.
        Calls _xy_routing function
        """

        if not isinstance( flit, TailFlit ):
            raise Exception("Error in Packet. Last flit in the packet is not a TailFlit. Cannot do routing.")

        header_flit_pointer = flit.get_header_pointer() # Flit here is a TailFlit

        next_hop_info       = self._xy_routing( header_flit_pointer )

        header_flit_pointer.update_routing_info( next_hop_info )
        self._debug_print( f"Routing packet in {buffer_name} to {next_hop_info}" )

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

        dest_id = header_flit.get_destination()
        
        dest_x, dest_y = self._get_pos_from_mapping( dest_id )

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


    def _get_pos_from_mapping(self, dest_id: int) -> tuple:
        """Returns the X and Y coordinates of the destination based on mapping."""
        for map in self._mapping_list:
            if map.task.task_id == dest_id:
                return map.assigned_pe
        raise Exception(f"Destination ID {dest_id} not found in the mapping list.")


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

    def get_pos(self) -> tuple:
        return (self._x, self._y)

    def is_active(self) -> bool:
        for buffer in self._input_buffers:
            if not buffer.is_empty():
                return True

        for buffer in self._output_buffers:
            if not buffer.is_empty():
                return True

        else:
            return False

    def _debug_print(self, string: str, with_tag: bool = True) -> None: 
        if self._debug_mode:
            if with_tag:
                print(f"{self} {string}")
            else:
                print(f"{string}")

    def __repr__(self):
        return f"[R({self._x}, {self._y})]"


if __name__ == "__main__":
    r""" 
    Condition:        

           P2
          /
   P1    R
    \    |
     R---R

    One packet sent from P1 to P2 (through 3 routers)
    """

    from .processing_element import ProcessingElement, TaskInfo, RequireInfo, TransmitInfo
    from .simulator import Map

    router_00       = Router( pos = (0, 0), debug_mode=True )
    router_10       = Router( pos = (1, 0), debug_mode=True )
    router_11       = Router( pos = (1, 1), debug_mode=True )
    router_lookup   = { (0, 0): router_00, (1, 0): router_10, (1, 1): router_11 }

    pe_00           = ProcessingElement( xy = (0, 0), debug_mode=True, router_lookup = router_lookup )
    pe_11           = ProcessingElement( xy = (1, 1), debug_mode=True, router_lookup = router_lookup )
    pe_lookup       = { (0, 0): pe_00, (1, 1): pe_11 }

    task_0          = TaskInfo(
                        task_id                     = 0, 
                        processing_cycles           = 4, 
                        expected_generated_packets  = 1, 
                        require_list                = [], 
                        is_transmit_task            = True, 
                        transmit_list               = [TransmitInfo(
                                                        id = 1,
                                                        require=1) ])


    task_1          = TaskInfo(
                        task_id                     = 1, 
                        processing_cycles           = 4, 
                        expected_generated_packets  = 1, 
                        require_list                = [RequireInfo(
                                                        require_type_id=0,
                                                        required_packets=1)], 
                        is_transmit_task            = False)


    mapping_list    = [ Map( task_0, (0, 0) ),
                        Map( task_1, (1, 1) )]

    for router in router_lookup.values():
        router.set_mapping_list( mapping_list )


    for mapping in mapping_list:
        pe = pe_lookup.get( mapping.assigned_pe )
        pe.assign_task( [mapping.task] )

    flit_list = []

    for i in range(40): 
        print(f"\n> {i}")
        pe_00.process(None)
        is_task_done = pe_11.process(None)

        if is_task_done:
            print(f"Application Done. Latency: {i}")
            break

        for router in router_lookup.values():
            flit_list = router.process( flit_list, router_lookup, pe_lookup )  



