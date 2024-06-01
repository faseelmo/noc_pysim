class Packet:
    def __init__(
        self,
        source_xy: tuple,
        dest_xy: tuple,
    ):
        self.payload_size = 4
        self.header_size = 1
        self.header_info = {
            "source": source_xy,
            "dest": dest_xy,
            "routing": [],
        }

    def __str__(self):
        return f"Packet: {self.header_info}"


if __name__ == "__main__":
    pass
