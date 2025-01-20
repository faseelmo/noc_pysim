from enum import Enum
from typing import Optional, Union
from collections import deque

import uuid 

from .flit import HeaderFlit, PayloadFlit, TailFlit, BufferLocation

class PacketStatus(Enum):
    IDLE            = "idle"
    TRANSMITTING    = "transmitting"
    ROUTING         = "routing"


class Packet:
    def __init__( self, source_xy: tuple, dest_id: Optional[int], source_task_id: int ):
        self._payload_size              = 2
        num_header_tail_flits           = 2

        self._source_task_id            = source_task_id

        self._size                      = self._payload_size + num_header_tail_flits 
        self._packet_content            = deque( maxlen=self._size )

        self._init_packet( self._packet_content, source_xy, dest_id, source_task_id )

        self._status                    = PacketStatus.IDLE
        self._pointer                   = 0

        self._flits_transmitted_count   = 0


    def _init_packet( self, packet_content: deque, source_xy: tuple , dest_id: Optional[int], source_task_id: int) -> None: 
        """ Initialize the packet with the header and payload information.
            "packet_content" is a member variable of the Packet class.
        """

        uid         = uuid.uuid4()

        header_flit = HeaderFlit( src_xy=source_xy, dest_id=dest_id, packet_uid=uid, source_task_id=source_task_id )
        packet_content.append(header_flit)

        for i in range(self._payload_size):
            packet_content.append( 
                            PayloadFlit( 
                                payload_index = i+1 , 
                                header_flit = header_flit) 
                            )

        packet_content.append( TailFlit( header_flit = header_flit ) )


    def pop_flit( self ) -> tuple[bool, Union[HeaderFlit, PayloadFlit, TailFlit]]: 
        """
        Returns a flit from the packet content based on the pointer index. 
        Also returns a flag indicating if the packet has been transmitted completely.
        """

        flit = self._packet_content[self._pointer]
        
        if isinstance( flit, TailFlit ): 
            self._pointer = 0
            return True, flit

        self._pointer += 1
        return False, flit


    def increment_flits(self) -> None:
        """Increment the number of flits transmitted by the packet."""
        if self._status is PacketStatus.IDLE and self._flits_transmitted_count == 0:
            self._flits_transmitted_count = 1
            self._status = PacketStatus.TRANSMITTING
        elif (
            self._status is PacketStatus.TRANSMITTING
            and self._flits_transmitted_count < self._size
        ):
            self._flits_transmitted_count += 1
        else:
            return

    def check_transmission_status(self) -> tuple[bool, Optional[int]]:
        """Check if the packet has been transmitted completely."""
        if (self._status is PacketStatus.TRANSMITTING) and (
            self._flits_transmitted_count == self._size
        ):
            self._status = PacketStatus.IDLE
            # print(f"{self} has been transmitted")
            self._flits_transmitted_count = 0
            return True, self._source_task_id
        else:
            return False, None

    def get_source_task_id(self) -> int:
        return self._source_task_id

    def get_flits_transmitted_count(self) -> int:
        return self._flits_transmitted_count

    def get_size(self) -> int:
        return self._size

    def get_status(self) -> PacketStatus:
        return self._status

    def get_uid(self) -> uuid.UUID:
        return self._packet_content[0].get_uid()
    

    def __str__(self) -> str:
        return (
            f"[Packet] Task id: {self._source_task_id}, "
            f"{self._packet_content[0].get_routing_info()} "
            # f"UID: {self._packet_content[0].get_uid()}"
        )


if __name__ == "__main__":
    packet = Packet(source_xy       =(0, 0), 
                    dest_id         =1, 
                    source_task_id  =0)

    print(f"{packet}\n")

    for i in range(packet._size + 1):

        is_transmitted, flit = packet.pop_flit()
        print(f"{flit}, pointer at {packet._pointer}")
