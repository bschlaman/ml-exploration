from __future__ import annotations
import math


class Vector2D:
    # TODO: extend a more generic Vector class
    def __init__(self, x: float, y: float):
        self.tail = [0, 0]
        self.x = x
        self.y = y

    def __str__(self):
        fp_precision = 5
        return str([round(self.x, 5), round(self.y, 5)])

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
        return self

    def sub(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError
        self.x -= other.x
        self.y -= other.y
        return self

    def dot_product(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError
        return self.x * other.x + self.y * other.y

    def to_tuple(self):
        return tuple(self.x, self.y)
