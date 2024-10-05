from src.processing_element import ProcessingElement, TaskInfo, RequireInfo
from src.router import Router

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

    flit_list   = []

    for i in range(40): 
        print(f"\n> {i}")
        pe_00.process(None)
        pe_10.process(None)
        is_task_done = pe_11.process(None)

        if is_task_done:
            print(f"Application Done. Latency: {i}")
            break

        for router in router_lookup.values():
            flit_list = router.process( flit_list, router_lookup, pe_lookup )  

    


test_router_pe_wait_in_input_buffer()