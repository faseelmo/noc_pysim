from enum import Enum


class PacketStatus(Enum):
    IDLE = "idle"
    TRANSMITTING = "transmitting"
    ROUTING = "routing"


class Packet:
    def __init__(self, source_xy: tuple, dest_xy: tuple, source_task_id: int):
        self.payload_size = 3
        self.header_size = 1
        self.header_info = {
            "source": source_xy,
            "dest": dest_xy,
            "routing": [],
        }
        self.size = self.payload_size + self.header_size
        self.source_task_id = source_task_id

        self.status = PacketStatus.IDLE
        self.flits_transmitted = 0
        self.current_location = None

    def update_location(self, object):
        """Update the current location of the packet.
        args: object (PE or Router)
        """
        self.current_location = object
        # print(f" ~ updating packet location to {object}")

    def increment_flits(self):
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

    def check_transmission_status(self):
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

    def __str__(self):
        return (
            f"Packet: {self.source_task_id} "
            # f"from {self.current_location},"
            f"Status: {self.status}, Flits Transmitted: {self.flits_transmitted}/{self.size}"
        )


if __name__ == "__main__":
    pass
