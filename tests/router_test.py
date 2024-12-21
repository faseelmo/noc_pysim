from src.router             import Router
from src.packet             import Packet
from src.flit               import TailFlit, HeaderFlit, PayloadFlit, EmptyFlit, BufferLocation
from src.processing_element import ProcessingElement, TaskInfo, RequireInfo, TransmitInfo
from src.simulator          import Map

from dataclasses import dataclass

@dataclass
class FakeTask:
    """Why? Because I dont want to create a whole new task for testing"""
    task_id: int

@dataclass 
class FakeMap:
    task: FakeTask
    assigned_pe: tuple[int, int]

def router_test_setup(pos_list: list[tuple[int, int]], debug_mode=False):    
    """Create a dictionary of routers with positions as keys"""
    router_lookup   = {}

    for pos in pos_list:
        router              = Router(pos, debug_mode=debug_mode)
        router_lookup[pos]  = router

    return router_lookup

def debug_print(string: str, debug_mode: bool):
    if debug_mode:
        print(string)

def test_router_pe_simple():
    """
    Condition: 
    PE(0,0) sends a packet to PE(1,1)
    """

    debug_mode = False

    router_00       = Router( pos = (0, 0), debug_mode=debug_mode )
    router_10       = Router( pos = (1, 0), debug_mode=debug_mode )
    router_11       = Router( pos = (1, 1), debug_mode=debug_mode )
    router_lookup   = { (0, 0): router_00, (1, 0): router_10, (1, 1): router_11 }

    pe_00           = ProcessingElement( xy = (0, 0), debug_mode=debug_mode, router_lookup = router_lookup )
    pe_11           = ProcessingElement( xy = (1, 1), debug_mode=debug_mode, router_lookup = router_lookup )
    pe_lookup       = { (0, 0): pe_00, (1, 1): pe_11 }

    task_0          = TaskInfo(
                        task_id                     = 0, 
                        processing_cycles           = 4, 
                        expected_generated_packets  = 1, 
                        require_list                = [], 
                        is_transmit_task            = True, 
                        transmit_list               = [TransmitInfo(
                                                        id=1, 
                                                        require=1) ] )


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

    for i in range(40): 
        debug_print(f"\n> {i}", debug_mode)
        pe_00.process(None)
        is_task_done = pe_11.process(None)

        if is_task_done:
            debug_print(f"Application Done. Latency: {i}", debug_mode)
            latency = i
            break

        for router in router_lookup.values():
            router._forward_output_and_process( router_lookup, pe_lookup )  

    debug_print(f"Latency: {latency}", debug_mode)
    assert latency == 35


def test_router_pe_wait_in_input_buffer(): 
    r"""
    Condition: 
    PE(0,0) and PE(1,0) send packets to PE(1,1)
    Packets from PE(1,0) will wait in the input buffer of Router(1,0)
    Graph structure:
    0
     \
      1
     /
    2
    Refer page 76 in thesis notes
    """

    debug_mode = False

    router_lookup = router_test_setup([(0,0), (1,0), (1,1)], debug_mode=debug_mode)

    # Packet Injection Task
    task_0  = TaskInfo(
                task_id                     = 0, 
                processing_cycles           = 4, 
                expected_generated_packets  = 1, 
                require_list                = [], 
                is_transmit_task            = True, 
                transmit_list               = [TransmitInfo(id=1, require=1)] )

    pe_00   = ProcessingElement( 
                xy              = (0, 0), 
                computing_list  = [ task_0 ], 
                debug_mode      = debug_mode, 
                router_lookup   = router_lookup )

    # Packet Injection Task
    task_2  = TaskInfo(
                task_id                     = 2, 
                processing_cycles           = 13, 
                expected_generated_packets  = 1, 
                require_list                = [], 
                is_transmit_task            = True, 
                transmit_list               = [TransmitInfo(id=1, require=1)] )

    pe_10   = ProcessingElement( 
                xy              = (1, 0), 
                computing_list  = [ task_2 ], 
                debug_mode      = debug_mode, 
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
                debug_mode      = debug_mode, 
                router_lookup   = router_lookup )

    pe_lookup   = { (0, 0): pe_00, (1, 0): pe_10, (1, 1): pe_11 }

    mapping_list = [ Map( task_0, (0, 0) ), 
                     Map( task_2, (1, 0) ),
                     Map( task_1, (1, 1) )]

    for router in router_lookup.values():
        router.set_mapping_list( mapping_list )

    for i in range(45): 

        debug_print(f"\n> {i}", debug_mode)
        pe_00.process(None)
        pe_10.process(None)
        is_task_done = pe_11.process(None)

        if is_task_done:
            debug_print(f"Application Done. Latency: {i}", debug_mode)
            latency = i 
            break

        for router in router_lookup.values():
            router._forward_output_and_process( router_lookup, pe_lookup )  

    assert latency == 39


def test_router_proper_in_out_buffer_1():
    """
    Condition:
    Intra Router Test
    Local buffer in R(0,0) inject with two packet with 
        2 empty slot(flits?) in between them. 
    Checking if FIFO is maintained for the second packet in the output buffer of the same router. 
    This check is performed by checking if the flits are getting into the buffer at the 
    expected clock cycle. Check page 36 of thesis notes for the diagram of the test case.
    """

    debug_mode = True

    router_00 = Router( (0, 0), debug_mode=debug_mode )
    router_10 = Router( (1, 0), debug_mode=debug_mode )

    router_lookup   = { (0, 0): router_00, (1, 0): router_10 }  

    mapping_list    = [FakeMap(FakeTask(task_id=0), assigned_pe=(0,0)), 
                       FakeMap(FakeTask(task_id=1), assigned_pe=(1,0)) ]

    packet_1        = Packet(source_xy     = (0, 0),
                           dest_id         = 1,
                           source_task_id  = 0)

    packet_2        = Packet(source_xy     = (0, 0),
                           dest_id         = 1,
                           source_task_id  = 0)

    for router in router_lookup.values():
        router.set_mapping_list( mapping_list )

    is_packet_1_injected = False
    is_packet_2_injected = False

    for i in range(15): 
        debug_print(f"\n> {i}", debug_mode)

        # if i == 7:
        #     peek_flit = router_00._local_input_buffer.peek()
        #     assert isinstance(peek_flit, TailFlit)


        # # Injecting Packet 1 
        if not is_packet_1_injected: 
            is_packet_1_transmitted, flit = packet_1.pop_flit()
            router_00._receive_flit( flit )

            if is_packet_1_transmitted:
                is_packet_1_injected = True

        # # Injecting Packet 2
        if i > 5:
            if not is_packet_2_injected: 
                is_packet_2_transmitted, flit = packet_2.pop_flit()
                router_00._receive_flit( flit )

                if is_packet_2_transmitted:
                    is_packet_2_injected = True


        for router in router_lookup.values():
            router._forward_output_and_process( router_lookup, {} )

        # Checking conditions. 
        # Refer page 37 of thesis notes
        if i == 9:
            peek_flit = router_00._east_output_buffer.queue[-1]
            debug_print(f"Peek Flit: {peek_flit}", debug_mode)
            assert isinstance(peek_flit, EmptyFlit), f"Expected EmptyFlit, got {peek_flit}"

        if i == 10:
            peek_flit = router_00._east_output_buffer.queue[-1]
            debug_print(f"Peek Flit: {peek_flit}", debug_mode)
            assert isinstance(peek_flit, HeaderFlit), f"Expected HeaderFlit in -1, got {peek_flit}"

        if i == 11:
            peek_flit = router_00._east_output_buffer.queue[-1]
            debug_print(f"Peek Flit: {peek_flit}", debug_mode)
            assert isinstance(peek_flit, PayloadFlit), f"Expected Payload in -1, got {peek_flit}"


# def test_router_proper_in_out_buffer_2():
#     """
#     Condition:
#     Inter Router Test
#     Local buffer in R(0,0) inject with two packet with 
#     2 empty slots in between. 
#     Checking if FIFO is maintained for the second packet in the output buffer of the same router. 
#     This check is performed by checking if the flits are getting into the buffer at the 
#     expected clock cycle. Check page 36 of thesis notes for the diagram of the test case.
#     """

#     router_00 = Router( (0, 0), debug_mode=False )
#     router_01 = Router( (0, 1), debug_mode=False )
#     router_10 = Router( (1, 0), debug_mode=False )

#     router_lookup = { (0, 0): router_00, (0, 1): router_01, (1, 0): router_10 }  

#     mapping_list    = [ FakeMap(FakeTask(task_id=2), (0,1)) ]

#     packet_1 = Packet(source_xy     =(0, 0),
#                     dest_id         =2,
#                     source_task_id  =0)

#     packet_2 = Packet(source_xy     =(0, 0),
#                     dest_id         =2,
#                     source_task_id  =0)

#     for router in router_lookup.values():
#         router.set_mapping_list( mapping_list )

#     is_packet_1_injected = False
#     is_packet_2_injected = False

#     for i in range(16): 
#         # print(f"\n> {i}")
#         flit_list = []

#         # Injecting Packet 1 
#         if not is_packet_1_injected: 
#             is_packet_1_transmitted, flit = packet_1.pop_flit()
#             flit_list.append(flit)

#             if is_packet_1_transmitted:
#                 is_packet_1_injected = True

#         # Injecting Packet 2
#         if i > 3:
#             if not is_packet_2_injected: 
#                 is_packet_2_transmitted, flit = packet_2.pop_flit()
#                 flit_list.append(flit)

#                 if is_packet_2_transmitted:
#                     is_packet_2_injected = True

#         if i < 8:
#             router_00._forward_output_and_process( flit_list, router_lookup, {} )

#         else: 

#             for router in router_lookup.values():
#                 flit_list = router._forward_output_and_process( flit_list, router_lookup, {} )

#         # Checking conditions.
#         if i == 10:
#             peek_flit = router_00._north_output_buffer.peek()
#             assert isinstance(peek_flit, TailFlit)

#             peek_flit = router_01._south_input_buffer.peek()
#             assert peek_flit is None

#         if i == 11:
#             peek_flit = router_00._north_output_buffer.peek()
#             assert isinstance(peek_flit, HeaderFlit)

#             peek_flit = router_01._south_input_buffer.peek()
#             assert isinstance(peek_flit, HeaderFlit)

#         if i == 14:
#             peek_flit = router_00._north_output_buffer.peek()
#             assert isinstance(peek_flit, PayloadFlit)

#             peek_flit = router_01._south_input_buffer.queue[-1]
#             assert isinstance(peek_flit, PayloadFlit)

#             peek_flit = router_01._south_input_buffer.peek()
#             assert isinstance(peek_flit, TailFlit)

#         if i == 15:
#             peek_flit = router_01._south_input_buffer.queue[-1]
#             assert isinstance(peek_flit, PayloadFlit)


def test_ready_at_the_same_time():
    """
    Condition:

    In R(1,0)
    South and West Buffer have packets that are ready to be transmitted at the same time
    to the North Output buffer. 

        - R(1,1) - 
            |
            | 
        - R(1,0) - 

    Buffer Scheduling, how does it work? 
    Fixed Priority apparently. Who woud've thunk right?
    If you are reading this. WHY?.

    FYI this contention resolving is also called "Arbitration logic".

    """
    debug_mode = True

    router_10 = Router( (1, 0), debug_mode=debug_mode )
    router_11 = Router( (1, 1), debug_mode=debug_mode )
    router_00 = Router( (0, 0), debug_mode=debug_mode )
    router_20 = Router( (2, 0), debug_mode=debug_mode )

    router_lookup = { (1, 0): router_10, 
                      (1, 1): router_11,
                      (0, 0): router_00,
                      (2, 0): router_20 }  

    mapping_list = [ FakeMap(FakeTask(task_id=1), (1,1)) ]

    for router in router_lookup.values():
        router.set_mapping_list( mapping_list )

    ## Packet 1
    packet_1 = Packet( source_xy       = (0, 0),
                       dest_id         = 1,
                       source_task_id  = 0 )

    ## Packet 2
    packet_2 = Packet( source_xy       = (0, 0),
                       dest_id         = 1,
                       source_task_id  = 0 )

    # Simulation 
    for i in range(16): 
        debug_print(f"\n> {i}", debug_mode)

        if i < 4: 
            flit_from_packet_1 = packet_1.pop_flit()[1]
            flit_from_packet_2 = packet_2.pop_flit()[1]
            router_00._receive_flit( flit_from_packet_1 )
            router_20._receive_flit( flit_from_packet_2 )

        for router in router_lookup.values():
            router._forward_output_and_process( router_lookup, {} )


        if i == 12: 
            tail_flit = router_10._north_output_buffer.queue[-1]    
            debug_print(f"Tail Flit at north output: {tail_flit}", debug_mode)
            assert isinstance(tail_flit, HeaderFlit)

            peek_flit = router_10._west_input_buffer.peek()    
            debug_print(f"Peek Flit at west input: {peek_flit}", debug_mode)
            assert isinstance(peek_flit, PayloadFlit)

            peek_flit = router_10._east_input_buffer.peek()    
            debug_print(f"Peek Flit at east input: {peek_flit}", debug_mode)
            assert isinstance(peek_flit, HeaderFlit)
    