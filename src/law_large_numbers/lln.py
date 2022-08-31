import random
from colorama import Fore
import logging
from statistics import mean
from utils.helpers import compact_repr

log = logging.getLogger(__name__)


def expected_value(vals: set[int], distribution="random") -> float:
    if distribution != "random":
        raise NotImplementedError("only distribution=random is currently supported")
    # for random distributions, expected value is the mean
    return mean(vals)


def lln_convergence():
    distribution = range(100)
    outcomes = []
    ev = expected_value(distribution)
    target_delta = 0.1
    log.info(f"{Fore.YELLOW}distribution{Fore.RESET}: {distribution}")
    log.info(f"{Fore.YELLOW}target delta{Fore.RESET}: {target_delta}")
    while abs(ev - mean(outcomes or [float("inf")])) > target_delta:
        outcomes.append(random.choice(distribution))
        log.info(f"{Fore.BLUE}expected val{Fore.RESET}:    {ev}")
        log.info(f"{Fore.BLUE}outcomes{Fore.RESET}:        {compact_repr(outcomes)}")
        log.info(f"{Fore.BLUE}outcomes (mean){Fore.RESET}: {mean(outcomes):.2f}")
        log.info(
            f"{Fore.BLUE}delta{Fore.RESET}:           {abs(ev - mean(outcomes)):.2f}"
        )
        input("press a key to continue")
