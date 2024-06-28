from src.processing_element import ProcessingElement, TaskInfo, RequireInfo
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
    packet = Packet(source_xy=(0, 0), dest_xy=(1, 1), source_task_id=packet_source)
    return [copy.deepcopy(packet) for _ in range(num_copies)]


def simulate(computing_list: list, packet_list: list):

    pe = ProcessingElement((0, 0), computing_list)
    current_packet = packet_list.pop(0)

    for cycle in range(MAX_CYCLES):
        print(f"\n> {cycle}")
        pe.process(current_packet)
        if not current_packet is None and current_packet.status is PacketStatus.IDLE:
            if len(packet_list) > 0:
                current_packet = packet_list.pop(0)
            else:
                current_packet = None
        print(f"POST: {pe}\t{pe_status_string(pe.compute_is_busy)}")
        if pe.check_task_requirements_met():
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

    latency = simulate(computing_list, packet_list)
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

    latency = simulate(computing_list, packet_list)
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

    latency = simulate(computing_list, packet_list)
    assert latency == 34

# test_task_sequential_1()
# test_task_sequential_2()
# test_task_parallel_3()