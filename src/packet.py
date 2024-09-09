from enum import Enum
from typing import Optional, Union
from collections import deque
from copy import deepcopy

import uuid 

from .flit import HeaderFlit, PayloadFlit, TailFlit, BufferLocation

class PacketStatus(Enum):
    IDLE            = "idle"
    TRANSMITTING    = "transmitting"
    ROUTING         = "routing"


class Packet:
    def __init__(self, source_xy: tuple, dest_xy: tuple, source_task_id: int):
        self.payload_size       = 2
        num_header_tail_flits   = 2

        self.size                   = self.payload_size + num_header_tail_flits 
        self.packet_content         = deque( maxlen=self.size )

        self._init_packet( self.packet_content, source_xy, dest_xy )

        self.status                 = PacketStatus.IDLE
        self.flits_transmitted      = 0
        self.source_task_id         = source_task_id
        self.packet_content_copy    = None


    def _init_packet(self, packet_content: deque, source_xy: tuple , dest_xy: tuple) -> None: 
        """ Initialize the packet with the header and payload information.
            "packet_content" is a member variable of the Packet class.
        """

        uid        = uuid.uuid4()

        header_flit = HeaderFlit( src_xy=source_xy, dest_xy=dest_xy, packet_uid=uid )
        packet_content.append(header_flit)

        for i in range(self.payload_size):
            packet_content.append( PayloadFlit( packet_uid = uid, payload_index = i+1 ) )

        packet_content.append( TailFlit( packet_uid = uid ) )


    def transmit_flit(self) -> tuple[bool, Union[dict, int]]: 
        """
        Creates a copy of the packet content and returns the first flit in the packet.
        Also returns a flag indicating if the packet has been transmitted completely.
        """

        is_transmitted_flag = False

        if self.packet_content_copy is None or len(self.packet_content_copy) == 0:
            # Creating a copy for the first time
            self.packet_content_copy = deepcopy(self.packet_content)

        flit = self.packet_content_copy.popleft()

        if isinstance(flit, TailFlit):
            is_transmitted_flag = True

        return is_transmitted_flag, flit


    def update_location(self, buffer_location: BufferLocation) -> None:
        """Update the current location of the packet.
        args: object (PE or Router)
        """
        self.current_buffer_loc = buffer_location

    def increment_flits(self) -> None:
        """Increment the number of flits transmitted by the packet."""
        if self.status is PacketStatus.IDLE and self.flits_transmitted == 0:
            self.flits_transmitted = 1
            self.status = PacketStatus.TRANSMITTING
        elif (
            self.status is PacketStatus.TRANSMITTING
            and self.flits_transmitted < self.size
        ):
            self.flits_transmitted += 1
        else:
            return

    def check_transmission_status(self) -> tuple[bool, Optional[int]]:
        """Check if the packet has been transmitted completely."""
        if (self.status is PacketStatus.TRANSMITTING) and (
            self.flits_transmitted == self.size
        ):
            self.status = PacketStatus.IDLE
            # print(f"{self} has been transmitted")
            self.flits_transmitted = 0
            return True, self.source_task_id
        else:
            return False, None

    def __str__(self) -> str:
        return (
            f"[Pakcet] Src: {self.source_task_id}, routing: {self.packet_content[0].get_routing_info()} "
        )


if __name__ == "__main__":
    packet = Packet(source_xy=(0, 0), 
                    dest_xy=(1, 1), 
                    source_task_id=0)

    print(f"{packet}\n")

    for i in range(packet.size + 1):

        is_transmitted, flit = packet.transmit_flit()
        print(f"{flit}, Packet content copy size: {len(packet.packet_content_copy)}")
