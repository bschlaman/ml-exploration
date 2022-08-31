from utils.math import minkowski_distance
import logging
from colorama import Fore

log = logging.getLogger(__name__)


def minkowski():
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
