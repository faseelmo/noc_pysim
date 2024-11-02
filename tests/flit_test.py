
from src.flit import HeaderFlit, PayloadFlit, TailFlit, BaseFlit

import uuid 

def test_compare(): 

    uid    = uuid.uuid4()

    header  = HeaderFlit( src_xy=(0, 0), dest_id=1, packet_uid=uid, source_task_id=0 )
    payload = PayloadFlit( payload_index=1, header_flit=header )
    tail    = TailFlit( header_flit=header )

    fake_uid    = uuid.uuid4()
    header_fake = HeaderFlit( src_xy=(0, 0), dest_id=1, packet_uid=fake_uid, source_task_id=0 )
    header_copy = HeaderFlit( src_xy=(0, 0), dest_id=1, packet_uid=uid, source_task_id=0 )

    assert header != header_fake, "Header Flit UUIDs should not be equal"
    assert header == header_copy, "Header Flit UUIDs should be equal"
    assert header == header_copy, "Header Flit UUIDs should be equal"

    payload_fake = PayloadFlit( payload_index=1, header_flit=header_fake )
    payload_copy = PayloadFlit( payload_index=1, header_flit=header_copy )

    assert payload  != payload_fake, "Payload Flit UUIDs should not be equal"
    assert header   != payload, "Header Flit UUIDs should not be equal to Payload Flit UUIDs"
    assert tail     != payload, "Tail Flit UUIDs should not be equal to Payload Flit UUIDs"
    assert payload  == payload_copy, "Payload Flit UUIDs should be equal to itself"

    tail_fake = TailFlit( header_flit=header_fake )
    tail_copy = TailFlit( header_flit=header_copy )

    assert tail     != tail_fake, "Tail Flit UUIDs should not be equal"
    assert header   != tail, "Header Flit UUIDs should not be equal to Tail Flit UUIDs"
    assert tail     == tail_copy, "Tail Flit UUIDs should be equal to itself"
    assert tail     != payload, "Tail Flit UUIDs should not be equal to Payload Flit UUIDs"