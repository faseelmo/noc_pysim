class Packet:
    def __init__(
        self,
        source_xy: tuple,
        dest_xy: tuple,
    ):
        self.payload_size = 3
        self.header_size = 1
        self.header_info = {
            "source": source_xy,
            "dest": dest_xy,
            "routing": [],
        }
        self.size = self.payload_size + self.header_size
        self.current_location = source_xy

    def update_location(self, location: tuple):
        self.current_location = location

    def __str__(self):
        return f"Packet: {self.header_info} in {self.current_location}"


if __name__ == "__main__":
    pass
