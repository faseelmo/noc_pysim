from src.buffer import Buffer
from src.packet import Packet
from src.flit import EmptyFlit, HeaderFlit, PayloadFlit, TailFlit

import pytest

def test_fill(): 

    buffer = Buffer(4)
    for flit in buffer.queue:
        assert isinstance(flit, EmptyFlit)  

    packet = Packet(source_xy=(0, 0), 
                    dest_id=1, 
                    source_task_id=0)

    assert buffer.can_do_routing()  == False
    assert buffer.is_full()         == False

    packet_size = packet.get_size()

    for _ in range(packet_size):
        _, flit = packet.pop_flit()
        buffer.add_flit(flit)   
        assert len(buffer.queue) == 4

    assert buffer.is_full()         == True
    assert buffer.can_do_routing()  == True


def test_remove_none(): 
    """
    Two flits are added and then 4 are removed. 
    The first two should be of type None because FIFO. 
    """

    buffer = Buffer(4)

    packet = Packet(source_xy=(0, 0), 
                    dest_id=1, 
                    source_task_id=0)

    for _ in range(2): 
        _, flit = packet.pop_flit()
        buffer.add_flit(flit)

    flit = buffer.remove()
    assert flit is None
    flit = buffer.remove()
    assert flit is None
    flit = buffer.remove()
    assert isinstance(flit, HeaderFlit)
    flit = buffer.remove()
    assert isinstance(flit, PayloadFlit)
    
def test_check_if_uid_is_removed():
    """
    A packet is completely added to the buffer and then removed.
    When the packet is fully removed, the UUID should be removed 
    from the buffer.
    """

    buffer = Buffer(4)
    packet = Packet(source_xy=(0, 0), 
                    dest_id=1, 
                    source_task_id=0)

    packet_size = packet.get_size()

    for _ in range(packet_size):
        _, flit = packet.pop_flit()
        buffer.add_flit(flit)   

        assert len(buffer.queue) == 4

    buffer.remove()    
    buffer.remove()
    buffer.remove()    
    buffer.remove()

    assert len(buffer._acceptable_flit_uids) == 0
    
def test_adding_new_packet_valid(): 
    """
    Conditions: 
    [  Header,  Payload,  Payload, Tail]
    -   In the buffer above two flits are removed and then 
        2 flits of a new packet are added.
    """

    buffer = Buffer(4)

    packet = Packet(source_xy=(0, 0), 
                    dest_id=1, 
                    source_task_id=0)

    packet_1_uid = packet.get_uid()
    packet_size = packet.get_size()

    for _ in range(packet_size):
        _, flit = packet.pop_flit()
        buffer.add_flit(flit)   

    packet_2 = Packet(source_xy=(0, 0), 
                           dest_id=1, 
                           source_task_id=0)

    packet_2_uid = packet_2.get_uid()

    _, flit_2 = packet_2.pop_flit()

    buffer.remove()
    buffer.add_flit(flit_2)

    buffer.remove()
    buffer.add_flit(flit_2)

    expected_uids = [packet_1_uid, packet_2_uid]

    for uid in buffer._acceptable_flit_uids:
        assert uid in expected_uids, f"Expected {uid} to be in {expected_uids}"


def test_adding_new_packet_invalid_1():
    """
    Conditions: Add flit (of new packet type) to the following buffers
    1. [  Empty,  Empty,  Header, Payload]
    """

    buffer = Buffer(4)

    packet = Packet(source_xy=(0, 0), 
                    dest_id=1, 
                    source_task_id=0)

    for _ in range(2):
        _, flit = packet.pop_flit()
        buffer.add_flit(flit)   

    packet_2 = Packet(source_xy=(0, 0), 
                           dest_id=1, 
                           source_task_id=0)

    _, flit_2 = packet_2.pop_flit()

    with pytest.raises(Exception, match="Cannot accept new packet and UUID not in acceptable list"):
        buffer.add_flit(flit_2)


def test_adding_new_packet_invalid_2():
    """
    Conditions: Add new flit to the following buffers
    2. [Payload, Header, Payload, Payload]
    """

    buffer = Buffer(4)

    packet = Packet(source_xy=(0, 0), 
                    dest_id=1, 
                    source_task_id=0)

    # Adding Packet 1 
    for _ in range(4):
        _, flit = packet.pop_flit()
        buffer.add_flit(flit)   

    packet_2 = Packet(source_xy=(0, 0), 
                      dest_id=1, 
                      source_task_id=0)

    # Popping 3 flits from the buffer
    buffer.remove()
    buffer.remove()
    buffer.remove()

    print(f"Buffer {buffer}")
    
    # Adding Packet 2 
    for _ in range(3):
        _, flit = packet_2.pop_flit()

        buffer.add_flit(flit)   
        print(f"Buffer {buffer}")

    # From Here
    packet_3 = Packet(source_xy=(0, 0),
                      dest_id=1,
                      source_task_id=0) 

    _, flit_3 = packet_3.pop_flit()

    with pytest.raises(Exception, match="Cannot add flit to full buffer."):
        buffer.add_flit(flit_3)

def test_adding_new_packet_invalid_3():
    """
    Conditions: Add new flit to the following buffers
    3. [Payload,   Tail,   Empty,   Empty]
    """

    buffer = Buffer(4)

    packet = Packet(source_xy=(0, 0), 
                    dest_id=1, 
                    source_task_id=0)

    for _ in range(4):
        _, flit = packet.pop_flit()

        buffer.add_flit(flit)   
        buffer.fill_emtpy_slots()

    buffer.remove()
    buffer.remove()
    buffer.fill_emtpy_slots()

    packet_2 = Packet(source_xy=(0, 0), 
                           dest_id=1, 
                           source_task_id=0)

    _, flit_2 = packet_2.pop_flit()

    with pytest.raises(Exception, match="Cannot add flit to full buffer."):
        buffer.add_flit(flit_2)


def test_adding_new_packet_valid_2(): 
    """
    Conditions: Add new flit to the following buffers
    3. [Payload, Payload,   Tail]
    """
    buffer = Buffer(4)

    packet = Packet(source_xy=(0, 0), 
                    dest_id=1, 
                    source_task_id=0)

    for _ in range(4):
        _, flit = packet.pop_flit()

        buffer.add_flit(flit)   
        buffer.fill_emtpy_slots()

    buffer.remove()

    packet_2 = Packet(source_xy=(0, 0), 
                           dest_id=1, 
                           source_task_id=0)

    _, flit_2 = packet_2.pop_flit()

    can_accept_flit = buffer.can_accept_flit(flit_2)
    assert can_accept_flit == True
    buffer.add_flit(flit_2)

