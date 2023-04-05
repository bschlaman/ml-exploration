# this file is used for mathematical utilities that I want to see represented
# as code.  there may be libraries for them, but I want to bootstrap a working
# learning tool with these functions
from __future__ import annotations

import math


def minkowski_distance(p1: tuple[float], p2: tuple[float], p: int) -> float:
    # TODO: (2022.08.29) show Chebyshev distance, i.e. p = ∞
    dimensionality = min(len(p1), len(p2))
    total = 0
    for n in range(dimensionality):
        total += pow(math.fabs(p1[n] - p2[n]), p)
    return pow(total, 1 / p)


def euclidean_distance(p1: tuple[float], p2: tuple[float]) -> float:
    return minkowski_distance(p1, p2, 2)


def manhattan_distance(p1: tuple[float], p2: tuple[float]) -> float:
    return minkowski_distance(p1, p2, 1)
