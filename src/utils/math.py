# this file is used for mathematical utilities that I want to see represented
# as code.  there may be libraries for them, but I want to bootstrap a working
# learning tool with these functions


def minkowski_distance(
    # TODO: (2022.08.29) show Chebyshev distance, i.e. p = ∞
    x_coords: tuple[float],
    y_coords: tuple[float],
    p: int = 2,
) -> float:
    dimensionality = min(len(x_coords), len(y_coords))
    total = 0
    for n in range(dimensionality):
        total += pow(abs(x_coords[n] - y_coords[n]), p)
    return pow(total, 1 / p)
