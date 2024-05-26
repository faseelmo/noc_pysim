from src.node import Node

class Router(Node):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __repr__(self):
        return f"R({self.x}, {self.y})"

if __name__ == "__main__":
    router = Router()