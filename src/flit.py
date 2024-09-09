import uuid
from enum import Enum


class BufferLocation(Enum):
    NORTH       = "north"
    WEST        = "west"
    SOUTH       = "south"
    EAST        = "east"
    LOCAL       = "local"
    UNASSIGNED  = "unassigned"

class HeaderFlit: 
    def __init__(self, src_xy: tuple, dest_xy: tuple, packet_uid: uuid.UUID): 
        self._src_xy         = src_xy
        self._dest_xy        = dest_xy
        self._packet_uid     = packet_uid

        self._routing_info   = []
        self._current_buffer = BufferLocation.UNASSIGNED

    def update_routing_info(self, routing_info: list) -> None:
        print( f"Updating routing information: {routing_info}" )
        self._routing_info = routing_info

    def update_current_buffer(self, buffer_location: BufferLocation) -> None:
        self._current_buffer = buffer_location

    def get_last_buffer(self) -> BufferLocation:
        return self._current_buffer

    def get_routing_info(self) -> list: 
        return self._routing_info

    def get_uid(self) -> uuid.UUID:
        return self._packet_uid


    def __str__(self): 
        return ( f"[Header Flit] {self._src_xy} -> {self._dest_xy}" )

class PayloadFlit: 
    def __init__(self, packet_uid: uuid.UUID, payload_index: int): 
        self._packet_uid     = packet_uid
        self._payload_index  = payload_index

    def __str__(self):
        return ( f"[Payload Flit] idx: {self._payload_index}" )

class TailFlit: 
    def __init__(self, packet_uid: uuid.UUID): 
        self._packet_uid = packet_uid

    def __str__(self):
        return ( f"[Tail Flit] UUID: {self._packet_uid}" )
