
class Router: 
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"R({self.x}, {self.y})"

if __name__ == "__main__":
    router = Router()