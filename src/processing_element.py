from packet import Packet
from enum import Enum


class PEStatus(Enum):
    PROCESSING = "processing"
    IDLE = "idle"
    DONE = "done"


class TXStatus(Enum):
    IDLE = "idle"
    DONE = "done"
    TRANSMITTING = "transmitting"


class ProcessingElement:
    def __init__(
        self,
        xy: tuple[int, int],
        require: int,
        generate: int,
        processing_time: int,
        packet_dest_list: list[dict],
    ):
        self.x_y = xy

        self.require = require
        self.generate = generate
        self.packet_dest_list = packet_dest_list

        self.current_processing_cycle = 0
        self.processing_cycle = processing_time

        self.pe_status = PEStatus.IDLE
        self.transmitting_status = TXStatus.IDLE

        self.require_count = 0
        self.generated_count = 0

        self.packet_size_in_flits = 0
        self.flits_transmitted_count = 0
        self.packet_transmitted_count = 0

        self.transmit_require = 0
        self.transmit_generated = 0
        self.current_transmit_packet = None

    def recieve_packets(self):
        """Simulate receiving a packet."""
        self.require_count += 1

    def validate_require_count(self):
        # In finalize_processing(), generated_count is incremented
        total_require_count = sum(
            packet_dest["require"] for packet_dest in self.packet_dest_list
        )
        if total_require_count > (self.generate - self.generated_count + 1):
            raise ValueError(
                f"Require needed: {total_require_count} != generate left: {(self.generate - self.generated_count)}"
            )

    def initiate_packet_transmission(self):
        if self.transmit_generated == self.transmit_require:
            if not self.packet_dest_list:
                print(f"No destination to transmit generated packets to")
                self.current_transmit_packet = None
                return
            print(f"First Transmission")
            self.transmit_generated = 0  # Resetting for new destination
            self.transmitting_status = TXStatus.TRANSMITTING
            packet_dest = self.packet_dest_list.pop()
            self.current_transmit_packet = Packet(self.x_y, packet_dest["pos"])
            self.transmit_require = packet_dest["require"]
            self.packet_size_in_flits = self.current_transmit_packet.size

        elif self.transmit_generated < self.transmit_require:
            print(f"Remaining Transmission")
            self.transmitting_status = TXStatus.TRANSMITTING
            self.packet_size_in_flits = self.current_transmit_packet.size

        else:
            print(
                f"transmit_generate: {self.transmit_generated}, transmit_require: {self.transmit_require}"
            )
            raise ValueError(f"Error in initiating packet transmission")

        print(
            f"\tTransmitting packet: {self.current_transmit_packet}, "
            f"require remaining: {self.transmit_require - self.transmit_generated}, "
            f"Size: {self.packet_size_in_flits}\n"
        )
        return

    def complete_packet_transmission(self):
        self.transmitting_status = TXStatus.DONE

        print(f"All Flits in the packet transmitted, " f"!Resetting!")

        self.flits_transmitted_count = 0
        self.packet_transmitted_count += 1
        self.packet_size_in_flits = 0

        self.transmit_generated += 1

    def transmit_packet(self):
        if (
            self.generate == self.generated_count
            and self.transmitting_status is TXStatus.IDLE
        ):
            if self.transmit_generated < self.transmit_require:
                self.initiate_packet_transmission()
            return  # Check position of return

        if self.generated_count > 0 and self.transmitting_status is TXStatus.IDLE:
            self.validate_require_count()
            self.initiate_packet_transmission()

        if self.flits_transmitted_count == self.packet_size_in_flits:
            self.complete_packet_transmission()
            return
            # return self.current_transmit_packet

        if self.transmitting_status is TXStatus.TRANSMITTING:
            self.flits_transmitted_count += 1

    def reset_processing_status(self):
        self.pe_status = PEStatus.IDLE
        self.current_processing_cycle = 0

    def complete_processing(self):
        self.pe_status = PEStatus.DONE
        self.current_processing_cycle = 0
        self.generated_count += 1
        if (
            self.transmitting_status is TXStatus.IDLE
            or self.transmitting_status is TXStatus.TRANSMITTING
        ):
            self.transmit_packet()

    def process(self):

        if self.require_count != self.require:
            print(f"Not all packets received")
            return

        if self.transmitting_status is TXStatus.TRANSMITTING:
            self.transmit_packet()

        if self.pe_status == PEStatus.DONE or self.generated_count >= self.generate:
            self.reset_processing_status()

        if (
            self.pe_status != PEStatus.PROCESSING
            and self.generated_count < self.generate
        ):
            self.pe_status = PEStatus.PROCESSING

        if self.current_processing_cycle >= self.processing_cycle:
            self.complete_processing()

        transmitted_packet = None
        if self.transmitting_status is TXStatus.DONE:
            transmitted_packet = self.current_transmit_packet
            self.transmitting_status = TXStatus.IDLE

        if self.pe_status == PEStatus.PROCESSING:
            self.current_processing_cycle += 1

        return transmitted_packet

    def __repr__(self):
        return f"PE({self.x_y})"


if __name__ == "__main__":
    pass
