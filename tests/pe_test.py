from src.processing_element import ProcessingElement, TaskInfo, RequireInfo, TransmitInfo
from src.packet import PacketStatus, Packet
import copy

MAX_CYCLES = 100


def pe_status_string(compute_is_busy: bool):
    if compute_is_busy:
        return "Computing"
    else:
        return "Free"


def check_if_pe_is_done(packet_list, pe):
    if len(packet_list) == 0 and not pe.compute_is_busy:
        return True


def create_packet_copies(num_copies, packet_source):
    packet = Packet(source_xy=(0, 0), dest_id=1, source_task_id=packet_source)
    return [copy.deepcopy(packet) for _ in range(num_copies)]


def simulate_pe(computing_list: list, packet_list: list):

    pe = ProcessingElement((0, 0), computing_list)
    current_packet = packet_list.pop(0)

    for cycle in range(MAX_CYCLES):
        # print(f"\n> {cycle}")
        pe.process(current_packet)
        if not current_packet is None and current_packet.get_status() is PacketStatus.IDLE:
            if len(packet_list) > 0:
                current_packet = packet_list.pop(0)
            else:
                current_packet = None
        # print(f"POST: {pe}\t{pe_status_string(pe.compute_is_busy)}")
        if pe._check_task_requirements_met():
            break

    return cycle


def test_task_sequential_1():
    """Test 1"""
    task_0 = TaskInfo(
        task_id=0,
        processing_cycles=5,
        expected_generated_packets=2,
        require_list=[
            RequireInfo(require_type_id=2, required_packets=3),
            RequireInfo(require_type_id=3, required_packets=2),
        ],
    )
    task_1 = TaskInfo(
        task_id=1,
        processing_cycles=4,
        expected_generated_packets=3,
        require_list=[
            RequireInfo(require_type_id=0, required_packets=2),
        ],
    )
    computing_list = [task_0, task_1]

    packet_2_copies = create_packet_copies(packet_source=2, num_copies=3)
    packet_3_copies = create_packet_copies(packet_source=3, num_copies=2)

    packet_list = [
        packet_2_copies.pop(0),
        packet_2_copies.pop(0),
        packet_3_copies.pop(0),
        packet_3_copies.pop(0),
        packet_2_copies.pop(0),
    ]

    latency = simulate_pe(computing_list, packet_list)
    assert latency == 42

def test_task_sequential_2():
    """Test 2"""
    task_0 = TaskInfo(
        task_id=0,
        processing_cycles=5,
        expected_generated_packets=2,
        require_list=[
            RequireInfo(require_type_id=2, required_packets=3),
            RequireInfo(require_type_id=3, required_packets=2),
        ],
    )
    task_1 = TaskInfo(
        task_id=1,
        processing_cycles=4,
        expected_generated_packets=3,
        require_list=[
            RequireInfo(require_type_id=0, required_packets=2),
            RequireInfo(require_type_id=4, required_packets=5),
        ],
    )
    computing_list = [task_0, task_1]

    packet_2_copies = create_packet_copies(packet_source=2, num_copies=3)
    packet_3_copies = create_packet_copies(packet_source=3, num_copies=2)
    packet_4_copies = create_packet_copies(packet_source=4, num_copies=5)

    packet_list = [
        packet_2_copies.pop(0),
        packet_2_copies.pop(0),
        packet_3_copies.pop(0),
        packet_3_copies.pop(0),
        packet_2_copies.pop(0),
        *packet_4_copies
    ]

    latency = simulate_pe(computing_list, packet_list)
    assert latency == 51

def test_task_parallel_3():
    """Test 3"""
    task_0 = TaskInfo(
        task_id=0,
        processing_cycles=7,
        expected_generated_packets=2,
        require_list=[
            RequireInfo(require_type_id=2, required_packets=1),
            RequireInfo(require_type_id=3, required_packets=1),
        ],
    )
    task_1 = TaskInfo(
        task_id=1,
        processing_cycles=4,
        expected_generated_packets=3,
        require_list=[
            RequireInfo(require_type_id=4, required_packets=2),
        ],
    )
    computing_list = [task_0, task_1]

    packet_2_copies = create_packet_copies(packet_source=2, num_copies=1)
    packet_3_copies = create_packet_copies(packet_source=3, num_copies=1)
    packet_4_copies = create_packet_copies(packet_source=4, num_copies=2)

    packet_list = [
        *packet_2_copies, 
        *packet_3_copies, 
        *packet_4_copies
    ]

    latency = simulate_pe(computing_list, packet_list)
    assert latency == 34

def test_pe_with_output_buffer(): 
    r"""
    Scenario: 
    - 1 & 3 are tasks assigned to PE. 0 and 2 are packets injected to PE.
    - 4 is the destination ID

    (0)
      \
       1 - 3 - (4)
      /
    (2)
    """

    task_1 = TaskInfo(
        task_id                     = 1, 
        processing_cycles           = 5, 
        expected_generated_packets  = 2, 
        require_list                = [RequireInfo(
                                        require_type_id=0, 
                                        required_packets=3), 
                                       RequireInfo(
                                        require_type_id=2, 
                                        required_packets=2)]
    )

    task_3                          = TaskInfo(
        task_id                     = 3,
        processing_cycles           = 4, 
        expected_generated_packets  = 3,
        require_list                = [RequireInfo(
                                        require_type_id=1, 
                                        required_packets=2)], 

        is_transmit_task            = True, 
        transmit_list               = [TransmitInfo(
                                        id=4,
                                        require=3) ] )

    packet_0 = Packet(
                source_xy       = (0, 0),
                dest_id         = 1,
                source_task_id  = 0)

    packet_2 = Packet(
                source_xy       = (0, 0),
                dest_id         = 1,
                source_task_id  = 2)

    packet_0_1 = copy.deepcopy(packet_0)
    packet_0_2 = copy.deepcopy(packet_0)
    packet_2_1 = copy.deepcopy(packet_2)

    packet_list     = [packet_0, packet_0_1,  packet_2, packet_2_1, packet_0_2,]
    computing_list  = [task_1, task_3]

    pe_1 = ProcessingElement((0, 0), computing_list, debug_mode=False)
  
    current_packet = packet_list.pop(0)
    for cycle in range(100):

        if cycle == 42 or cycle == 53 or cycle == 60: 
            pe_1._empty_output_buffer(task_3)

        pe_1.process(current_packet)

        # Injecting Packets 
        if not current_packet is None and current_packet.get_status() is PacketStatus.IDLE:
            if len(packet_list) > 0: current_packet = packet_list.pop(0)
            else: current_packet = None

        if pe_1._check_task_requirements_met():
            latency = cycle
            break

    assert latency == 60

def test_pe_with_in_out_buffer(): 
    r"""
    Scenario: 
    - 1 & 3 are tasks assigned to PE. 0 and 2 are packets injected to PE.
    - 4 is the destination ID

    0 
     \
      1 - 3 - 4  
     /
    2 
    """

    task_1 = TaskInfo(
        task_id                     = 1, 
        processing_cycles           = 5, 
        expected_generated_packets  = 2, 
        require_list                = [RequireInfo(
                                        require_type_id=0, 
                                        required_packets=3), 
                                       RequireInfo(
                                        require_type_id=2, 
                                        required_packets=2)]
    )

    task_3                          = TaskInfo(
        task_id                     = 3,
        processing_cycles           = 4, 
        expected_generated_packets  = 3,
        require_list                = [RequireInfo(
                                        require_type_id=1, 
                                        required_packets=2)], 

        is_transmit_task            = True, 
        transmit_list                 = [TransmitInfo(
                                        id=4,
                                        require=3) ] )
    

    packet_0 = Packet(
                source_xy       = (0, 0),
                dest_id         = 1,
                source_task_id  = 0)

    packet_2 = Packet(
                source_xy       = (0, 0),
                dest_id         = (1, 1),
                source_task_id  = 2)

    packet_0_1 = copy.deepcopy( packet_0 )
    packet_0_2 = copy.deepcopy( packet_0 )
    packet_2_1 = copy.deepcopy( packet_2 )

    packet_list     = [packet_0, packet_0_1,  packet_2, packet_2_1, packet_0_2,]
    computing_list  = [task_1, task_3]

    pe_1 = ProcessingElement( (0, 0), computing_list, debug_mode=False )
    current_packet = packet_list.pop(0)

    is_packet_transmitted = False

    for cycle in range(100):
        # print( f"\n> {cycle}" )

        # Start Injecting Flits
        if not is_packet_transmitted:
            is_flit_transmitted, flit = current_packet.pop_flit()

            # print( f"{flit} task_id: {flit.get_source_task_id()}" ) 
            # print(f"Injecting Flit: {flit} to buffer {pe_1.input_network_interface}")
            pe_1.receive_flits( flit ) # Injecting Flit to PE

            if is_flit_transmitted: 
                if len( packet_list ) > 0: 
                    current_packet = packet_list.pop(0)

                else: 
                    is_packet_transmitted = True
        # End Injecting Flits

        if cycle == 42 or cycle == 53 or cycle == 60: 
            pe_1._empty_output_buffer(task_3)

        pe_1.process(None)

        if pe_1._check_task_requirements_met():
            latency = cycle
            break

    assert latency == 60

def test_injection_pe(): 
    """
    Condition: 
    PE Injecting one packet.  
    But the ouput buffer is full.
    """

    task_0                          = TaskInfo(
        task_id                     = 0,
        processing_cycles           = 4, 
        expected_generated_packets  = 1,
        require_list                = [], 
        is_transmit_task            = True, 
        transmit_list               = [TransmitInfo(id=1, require=1)]
    )


    pe = ProcessingElement( (0, 0), [task_0], debug_mode=False )

    for cycle in range(20): 
        # print( f"\n> {cycle}" )
        done_processing = pe.process(None)

        if done_processing: 
            latency = cycle
            break 

        if cycle == 5: 
            pe._empty_output_buffer(task_0)

    assert latency == 6


def test_injection_pe_3_packets(): 
    """
    Condition: 
    PE Injecting three packet.  
    But the ouput buffer is full.
    """

    task_0                          = TaskInfo(
        task_id                     = 0,
        processing_cycles           = 4, 
        expected_generated_packets  = 3,
        require_list                = [], 
        is_transmit_task            = True, 
        transmit_list               = [TransmitInfo(id=1, require=3)]   
    )


    pe = ProcessingElement( (0, 0), [task_0], debug_mode=False )

    for cycle in range(40): 
        # print( f"\n> {cycle}" )
        done_processing = pe.process(None)

        if done_processing: 
            latency = cycle
            break 

        if cycle == 10 or cycle == 20 or cycle == 30: 
            pe._empty_output_buffer(task_0)

    assert latency == 31


def test_transmitting_to_different_pe(): 
    """
    Condition, 
    Terminal (transmit) node is sending packets to two different PEs. 
    """
    task_0  = TaskInfo(
            task_id                     = 0,
            processing_cycles           = 3, 
            expected_generated_packets  = 5,
            require_list                = [],
            is_transmit_task            = True, 
            transmit_list               = [ TransmitInfo(
                                                id = 1, 
                                                require = 2), 
                                            TransmitInfo(
                                                id = 2,
                                                require = 3) ] ) 

    pe = ProcessingElement( (0, 0), [task_0], debug_mode=False )

    for cycle in range(20): 
        # print( f"\n> {cycle}" )
        done_processing = pe.process(None)

        if done_processing: 
            latency = cycle
            break 

        if cycle == 4 or cycle == 10 : 
            pe._empty_output_buffer(task_0)

test_transmitting_to_different_pe()