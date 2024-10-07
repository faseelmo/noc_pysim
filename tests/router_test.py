from src.processing_element import ProcessingElement, TaskInfo, RequireInfo
from src.router import Router
from src.packet import Packet

def router_test_setup(pos_list: list[tuple[int, int]]): 
    """Create a dictionary of routers with positions as keys"""
    router_lookup   = {}

    for pos in pos_list:
        router              = Router(pos, debug_mode=True)
        router_lookup[pos]  = router

    return router_lookup


def test_router_pe_simple():
    """
    Condition: 
    PE(0,0) sends a packet to PE(1,1)
    """
    router_lookup = router_test_setup([(0,0), (1,0), (1,1)])

    task_0  = TaskInfo(
                task_id                     = 0, 
                processing_cycles           = 4, 
                expected_generated_packets  = 1, 
                require_list                = [], 
                is_transmit_task            = True, 
                transmit_dest_xy            = (1, 1)
            )

    pe_00   = ProcessingElement( 
                xy              = (0, 0), 
                computing_list  = [ task_0 ], 
                debug_mode      = True, 
                router_lookup   = router_lookup )

    task_1  = TaskInfo(
                task_id                     = 1, 
                processing_cycles           = 4, 
                expected_generated_packets  = 1, 
                require_list                = [RequireInfo(
                                                require_type_id=0,
                                                required_packets=1)], 
                is_transmit_task            = False, 
            )

    pe_11   = ProcessingElement( 
                xy              = (1, 1), 
                computing_list  = [ task_1 ], 
                debug_mode      = True, 
                router_lookup   = router_lookup )

    pe_lookup   = { (0, 0): pe_00, (1, 1): pe_11 }

    latency     = 0
    flit_list   = []
    
    for i in range(40): 
        pe_00.process(None)

        is_application_done = pe_11.process(None)

        if is_application_done: 
            latency = i
            break

        for router in router_lookup.values():
            flit_list = router.process( flit_list, router_lookup, pe_lookup )  

    assert latency == 36

def test_router_pe_wait_in_input_buffer(): 
    """
    Condition: 
    PE(0,0) and PE(1,0) send packets to PE(1,1)
    Packets from PE(1,0) will wait in the input buffer of Router(1,0)
    Graph structure:
    0
     \1
      1
     /
    2
    """
    router_lookup = router_test_setup([(0,0), (1,0), (1,1)])

    # Packet Injection Task
    task_0  = TaskInfo(
                task_id                     = 0, 
                processing_cycles           = 4, 
                expected_generated_packets  = 1, 
                require_list                = [], 
                is_transmit_task            = True, 
                transmit_dest_xy            = (1, 1)
            )

    pe_00   = ProcessingElement( 
                xy              = (0, 0), 
                computing_list  = [ task_0 ], 
                debug_mode      = True, 
                router_lookup   = router_lookup )

    # Packet Injection Task
    task_2  = TaskInfo(
                task_id                     = 2, 
                processing_cycles           = 13, 
                expected_generated_packets  = 1, 
                require_list                = [], 
                is_transmit_task            = True, 
                transmit_dest_xy            = (1, 1)
            )

    pe_10   = ProcessingElement( 
                xy              = (1, 0), 
                computing_list  = [ task_2 ], 
                debug_mode      = True, 
                router_lookup   = router_lookup )

    task_1  = TaskInfo(
                task_id                     = 1, 
                processing_cycles           = 4, 
                expected_generated_packets  = 1, 
                require_list                = [RequireInfo(
                                                require_type_id=0,
                                                required_packets=1), 
                                               RequireInfo(
                                                require_type_id=2,
                                                required_packets=1)],
                is_transmit_task            = False, 
            )

    pe_11   = ProcessingElement( 
                xy              = (1, 1), 
                computing_list  = [ task_1 ], 
                debug_mode      = True, 
                router_lookup   = router_lookup )

    pe_lookup   = { (0, 0): pe_00, (1, 0): pe_10, (1, 1): pe_11 }



    for i in range(45): 
        flit_list = []

        print(f"\n> {i}")
        pe_00.process(None)
        pe_10.process(None)
        is_task_done = pe_11.process(None)

        if is_task_done:
            print(f"Application Done. Latency: {i}")
            latency = i 
            break

        for router in router_lookup.values():
            flit_list = router.process( flit_list, router_lookup, pe_lookup )  

        # for router in router_lookup.values():
        #     new_flit_list = router._forward_output_buffer_flits( router_lookup, pe_lookup )
        #     flit_list.extend(new_flit_list)



        # for router in router_lookup.values():
        #     router._forward_input_buffer_flits()

        # for router in router_lookup.values():

        #     for flit in flit_list:
        #         router._receive_flit( flit )

        # for router in router_lookup.values(): 
        #     router.management()

    assert latency == 40

# test_router_pe_wait_in_input_buffer() 


def test_router_proper_in_out_buffer():
    """
    Condition:
    Router gets packet in the local input buffer. 
    Gets routed to other buffer. 
    The same local input buffer receives packets with a 3 buffer gap
    from the last packet's tail. 
    See if FIFO is maintained
    """

    router = Router( (0, 0), debug_mode=True )
    router_lookup = { (0, 0): router }  

    packet_1 = Packet(source_xy     =(0, 0),
                    dest_xy         =(0,1),
                    source_task_id  =0)

    packet_2 = Packet(source_xy     =(0, 0),
                    dest_xy         =(1,0),
                    source_task_id  =0)


    is_packet_1_injected = False

    for i in range(8): 
        print(f"\n> {i}")
        flit_list = []

        if i == 6: 
            print(f"Adding Packet 2")
            flit_list.append(packet_2.pop_flit()[1])
            print(f"{router._local_input_buffer}") 

        # Injecting Packet
        if not is_packet_1_injected: 
            is_flit_transmitted, flit = packet_1.pop_flit()
            flit_list.append(flit)

            if is_flit_transmitted:
                is_packet_1_injected = True

            

        # Skipping not finding the next Router or PE
        try: 
            router.process(flit_list, router_lookup, {})
        except AttributeError as e: 
            continue


# test_router_proper_in_out_buffer()