from __future__ import annotations

import math

from utils.printing import compact_repr


class Vector:
    CARTESIAN_NOTATION_MAP = {
        "x": 0,
        "y": 1,
        "z": 2,
        "w": 3,
    }

    def __init__(self, *components: float):
        if not components:
            raise Exception("must supply components")
        self.__dict__["components"] = list(components)

    def __str__(self):
        return compact_repr(self.components)

    def magnitude(self):
        return math.sqrt(sum(map(lambda _: _**2, self.components)))

    def normalize(self):
        mag = self.magnitude()
        for i in range(len(self.components)):
            self.components[i] /= mag

    def scale(self, scalar: float):
        for _ in range(len(self.components)):
            self.components[_] *= scalar

    def add(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError
        for i, c in enumerate(other.components):
            if i < len(self.components):
                self.components[i] += c
            else:
                self.components.append(c)

    def sub(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError
        for i, c in enumerate(other.components):
            if i < len(self.components):
                self.components[i] -= c
            else:
                self.components.append(-c)

    def dot_product(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError
        lesser_dimensionality = min(len(self.components), len(other.components))
        return sum(
            self.components[i] * other.components[i]
            for i in range(lesser_dimensionality)
        )

    # TODO (2022.09.07) I am currently undecided on what I want the behavior to be for components that
    # are missing.  Should I simply default to 0, or is dimensionality completely fixed at
    # instantiation?  If that's the case, should I even allow adding vectors with different
    # dimensionalities?
    def __getattr__(self, attr):
        if attr not in self.__class__.CARTESIAN_NOTATION_MAP:
            raise AttributeError(
                f"invalid component: {attr}, choose from"
                f" {','.join(self.__class__.CARTESIAN_NOTATION_MAP.keys())}"
            )
        if self.__class__.CARTESIAN_NOTATION_MAP[attr] >= len(self.components):
            raise AttributeError(f"component dimensionality too high: {attr}")
        return self.components[self.__class__.CARTESIAN_NOTATION_MAP[attr]]
        # return getattr(self.component, attr)

    def __setattr__(self, attr, val):
        if attr not in self.__class__.CARTESIAN_NOTATION_MAP:
            raise AttributeError(
                f"invalid component: {attr}, choose from"
                f" {','.join(self.__class__.CARTESIAN_NOTATION_MAP.keys())}"
            )
        if self.__class__.CARTESIAN_NOTATION_MAP[attr] >= len(self.components):
            raise AttributeError(f"component dimensionality too high: {attr}")
        if self.components and type(val) != type(self.components[0]):
            raise TypeError
        self.components[self.__class__.CARTESIAN_NOTATION_MAP[attr]] = val


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
        return (self.x, self.y)
