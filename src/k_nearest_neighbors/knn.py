import logging
import random
from itertools import chain, combinations
from math import sqrt
from statistics import mean

from colorama import Fore

from utils.helpers import compact_repr, data_print
from utils.math.distance import euclidean_distance, minkowski_distance

log = logging.getLogger(__name__)


def two_point_distance_increase() -> None:
    for d in range(1, 10):
        log.info(f"{Fore.YELLOW}dimensionality{Fore.RESET}: {d=}")
        p1 = tuple(random.randrange(-99, 100) for _ in range(d))
        p2 = tuple(random.randrange(-99, 100) for _ in range(d))
        log.info(
            f"{Fore.YELLOW}using points{Fore.RESET}: "
            f"p1: {compact_repr(p1)}, p2: {compact_repr(p2)}"
        )
        ed = euclidean_distance(p1, p2)
        log.info(f"{Fore.BLUE}euclidean distance{Fore.RESET}: {ed:.5f}")


def curse_of_dimensionality() -> None:
    """Plot 100 points randomly and find the average distance
    both to the origin, and to all the other points"""
    num_points = 100
    max_coord = 100
    for d in chain(range(1, 5), range(96, 100)):
        labeled_data = {
            "dimensionality": f"{d=}",
            "num points": num_points,
            "coord range": str([-max_coord, max_coord]),  # brackets notation
        }
        for line in data_print(labeled_data, Fore.YELLOW):
            log.info(line)

        origin = tuple(0 for _ in range(d))
        points = tuple(
            tuple(random.randrange(-max_coord, max_coord + 1) for _ in range(d))
            for _ in range(num_points)
        )
        avg_dist_to_origin = mean(
            [euclidean_distance(origin, point) for point in points]
        )
        avg_dist_pairwise = mean(
            [euclidean_distance(p1, p2) for p1, p2 in combinations(points, 2)]
        )

        labeled_data = {
            "avg dist to origin": avg_dist_to_origin,
            "avg dist pairwise": avg_dist_pairwise,
        }
        for line in data_print(labeled_data, Fore.BLUE):
            log.info(line)

        # compare the closest and farthest points from p0
        # in the future, I could also track the variance in distances
        # or plot a histogram of the distances to show they converge
        # arount a certain value
        p0 = points[0]
        closest_neighbor = min(points[1:], key=lambda p: euclidean_distance(p0, p))
        farthest_neighbor = max(points[1:], key=lambda p: euclidean_distance(p0, p))
        closest_neighbor_dist = euclidean_distance(p0, closest_neighbor)
        farthest_neighbor_dist = euclidean_distance(p0, farthest_neighbor)
        max_possible_distance = 2 * sqrt(d * (max_coord**2))
        labeled_data = {
            "point chosen": compact_repr(p0),
            "dist to closest neighbor": closest_neighbor_dist,
            "dist to farthest neighbor": farthest_neighbor_dist,
            "max possible distance": max_possible_distance,
            "farthest and nearest differential": (
                farthest_neighbor_dist - closest_neighbor_dist
            )
            / max_possible_distance,
        }
        for line in data_print(labeled_data, Fore.MAGENTA):
            log.info(line)


def minkowski() -> None:
    p1, p2 = (3, 5), (6, -1)

    p = 2
    log.info(f"{Fore.YELLOW}testing minkowski with points{Fore.RESET}: {p1=}, {p2=}")
    log.info(f"{Fore.YELLOW}using p{Fore.RESET}: {p}")
    res = minkowski_distance(p1, p2, p)
    log.info(f"{Fore.BLUE}result{Fore.RESET}: {res:.5f}")

    p = 1
    log.info(f"{Fore.YELLOW}using p{Fore.RESET}: {p}")
    res = minkowski_distance(p1, p2, p)
    log.info(f"{Fore.BLUE}result{Fore.RESET}: {res:.5f}")
