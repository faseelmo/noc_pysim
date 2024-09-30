from src.buffer import Buffer
from src.packet import Packet
from src.flit import EmptyFlit

def test_fill_and_empty(): 

    buffer = Buffer(4)
    for flit in buffer.queue:
        assert isinstance(flit, EmptyFlit)  

    packet = Packet(source_xy=(0, 0), 
                    dest_xy=(1, 1), 
                    source_task_id=0)

    assert buffer.can_do_routing()  == False
    assert buffer.is_full()        == False

    packet_size = packet.get_size()
    for _ in range(packet_size):
        _, flit = packet.transmit_flit()
        buffer.add_flit(flit)   
        buffer.fill_with_empty_flits()
        assert len(buffer.queue) == 4

    assert buffer.is_full()        == True
    assert buffer.can_do_routing()  == True

def test_return_none(): 

    buffer = Buffer(4)
    for flit in buffer.queue:
        assert isinstance(flit, EmptyFlit)  

    packet = Packet(source_xy=(0, 0), 
                    dest_xy=(1, 1), 
                    source_task_id=0)

    for _ in range(2): 
        _, flit = packet.transmit_flit()
        buffer.add_flit(flit)
        buffer.fill_with_empty_flits()
        assert len(buffer.queue) == 4

    assert buffer.can_do_routing()  == False
    assert buffer.is_full()        == False

    flit = buffer.remove()
    assert flit is None
    flit = buffer.remove()
    assert flit is None
    
    
