from enum import Enum
from typing import Optional, Union
from collections import deque
from copy import deepcopy


class PacketStatus(Enum):
    IDLE            = "idle"
    TRANSMITTING    = "transmitting"
    ROUTING         = "routing"


class Packet:
    def __init__(self, source_xy: tuple, dest_xy: tuple, source_task_id: int):
        payload_size    = 3
        header_size     = 1

        self.status                 = PacketStatus.IDLE
        self.flits_transmitted      = 0
        self.source_task_id         = source_task_id
        self.current_location       = None
        self.packet_content_copy    = None

        self.size           = payload_size + header_size
        self.packet_content = deque(maxlen=self.size)
        self._init_packet(self.packet_content, source_xy, dest_xy)

    def _init_packet(self, packet_content: deque, source_xy: tuple , dest_xy: tuple) -> None: 
        """Initialize the packet with the header and payload information."""
        header_info = {"source": source_xy,
                       "dest": dest_xy,
                       "routing": []}

        packet_content.append(header_info)

        for i in range(self.size - 1):
            packet_content.append(i+1)


    def get_flit(self) -> tuple[bool, Union[dict, int]]: 
        """
        Creates a copy of the packet content and returns the first flit in the packet.
        Also returns a flag indicating if the packet has been transmitted completely.
        """

        is_transmitted_flag = False

        if self.packet_content_copy is None or len(self.packet_content_copy) == 0:
            self.packet_content_copy = deepcopy(self.packet_content)

        flit = self.packet_content_copy.popleft()

        if len(self.packet_content_copy) == 0:
            is_transmitted_flag = True

        return is_transmitted_flag, flit


    def update_location(self, object) -> None:
        """Update the current location of the packet.
        args: object (PE or Router)
        """
        self.current_location = object
        # print(f" ~ updating packet location to {object}")

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
            f"Packet: {self.source_task_id} "
        )


if __name__ == "__main__":
    packet = Packet(source_xy=(0, 0), 
                    dest_xy=(1, 1), 
                    source_task_id=0)

    for i in range(packet.size + 1):
        print(f"{packet.get_flit()}, len: {len(packet.packet_content_copy)}")
