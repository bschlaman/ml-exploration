from utils.math import minkowski_distance, euclidean_distance
from utils.helpers import compact_repr
import logging
import random
from colorama import Fore

log = logging.getLogger(__name__)


def curse_of_dimensionality() -> None:
    num_features = 10
    for d in range(1, 190):
        log.info(f"{Fore.YELLOW}dimensionality{Fore.RESET}: {d=}")
        p1 = tuple(random.randrange(-100, 100) for _ in range(d))
        p2 = tuple(random.randrange(-100, 100) for _ in range(d))
        log.info(
            f"{Fore.YELLOW}using points{Fore.RESET}: "
            f"p1: {compact_repr(p1)}, p2: {compact_repr(p2)}"
        )
        ed = euclidean_distance(p1, p2)
        log.info(f"{Fore.BLUE}euclidean distance{Fore.RESET}: {ed:.5f}")


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
