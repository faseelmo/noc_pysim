import uuid
import copy
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
        self.src_xy         = src_xy
        self.dest_xy        = dest_xy
        self.packet_uid     = packet_uid

        self._routing_info   = []
        self._current_buffer = BufferLocation.UNASSIGNED

    def update_routing_info(self, routing_info: list) -> None:
        self._routing_info = routing_info

    def get_last_buffer(self) -> BufferLocation:
        return self._current_buffer

    def get_routing_info(self) -> list: 
        return self._routing_info

    def update_current_buffer(self, buffer_location: BufferLocation) -> None:
        self._current_buffer = buffer_location

    def __deepcopy__(self, memo):
        new_copy = HeaderFlit(
            src_xy=copy.deepcopy(self.src_xy, memo),
            dest_xy=copy.deepcopy(self.dest_xy, memo),
            packet_uid=self.packet_uid  # UUIDs are immutable, so no need to deepcopy
        )
        new_copy._routing_info = copy.deepcopy(self._routing_info, memo)
        new_copy._current_buffer = copy.deepcopy(self._current_buffer, memo)
        return new_copy

    def __str__(self): 
        return ( f"[Header Flit] ({self.packet_uid}) {self.src_xy} -> {self.dest_xy}" )

class PayloadFlit: 
    def __init__(self, packet_uid: uuid.UUID, payload_index: int): 
        self.packet_uid     = packet_uid
        self.payload_index  = payload_index

    def __deepcopy__(self, memo):
        return PayloadFlit(
            packet_uid      = self.packet_uid,  
            payload_index   = self.payload_index
        )

    def __str__(self):
        return ( f"[Payload Flit] ({self.packet_uid}) Payload {self.payload_index}" )

class TailFlit: 
    def __init__(self, packet_uid: uuid.UUID): 
        self.packet_uid = packet_uid

    def __deepcopy__(self, memo):
        return TailFlit(
            packet_uid=self.packet_uid  
        )

    def __str__(self):
        return ( f"[Tail Flit] ({self.packet_uid})" )
