from utils.math import minkowski_distance, euclidean_distance
from utils.helpers import compact_repr, data_print
import logging
import random
from colorama import Fore
from statistics import mean
from itertools import combinations

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
    for d in range(1, 10):
        labeled_data = {
            "dimensionality": f"{d=}",
            "num points": num_points,
        }
        for line in data_print(labeled_data, Fore.YELLOW):
            log.info(line)

        origin = tuple(0 for _ in range(d))
        points = tuple(
            tuple(random.randrange(-99, 100) for _ in range(d))
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
