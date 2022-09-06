# this file is used for mathematical utilities that I want to see represented
# as code.  there may be libraries for them, but I want to bootstrap a working
# learning tool with these functions
import math


def minkowski_distance(p1: tuple[float], p2: tuple[float], p: int) -> float:
    # TODO: (2022.08.29) show Chebyshev distance, i.e. p = ∞
    dimensionality = min(len(p1), len(p2))
    total = 0
    for n in range(dimensionality):
        total += pow(abs(p1[n] - p2[n]), p)
    return pow(total, 1 / p)


def euclidean_distance(p1: tuple[float], p2: tuple[float], p: int = 2) -> float:
    return minkowski_distance(p1, p2, 2)


def manhattan_distance(p1: tuple[float], p2: tuple[float], p: int = 2) -> float:
    return minkowski_distance(p1, p2, 1)


class Vector2D:
    # TODO: extend a more generic Vector class
    def __init__(self, x: float, y: float):
        self.tail = [0, 0]
        self.x = x
        self.y = y

    def __str__(self):
        return str([self.x, self.y])

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        mag = self.magnitude()
        self.x /= mag
        self.y /= mag

    def scale(self, scalar: float):
        self.x *= scalar
        self.y *= scalar

    # def set_offset(self, dx: float, dy: float):
    #     self.tail = [dx, dy]
    #     self.add(Vector2D(dx, dy))

    def add(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError
        self.x += other.x
        self.y += other.y

    def dot_product(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError
        return self.x * other.x + self.y * other.y

    def to_tuple(self):
        return tuple(self.x, self.y)
