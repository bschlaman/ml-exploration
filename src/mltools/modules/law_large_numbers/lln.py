import logging
import random
from statistics import mean

from bpyutils.formatting import std

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
    labeled_data = {
        "distribution": distribution,
        "target delta": target_delta,
    }
    for line in std.data_print(labeled_data):
        log.info(line)
    while abs(ev - mean(outcomes or [float("inf")])) > target_delta:
        outcomes.append(random.choice(distribution))
        labeled_data = {
            "expected val": ev,
            "outcomes": std.compact_repr(outcomes),
            "outcomes (mean)": mean(outcomes),
            "delta": abs(ev - mean(outcomes)),
        }
        for line in std.data_print(labeled_data):
            log.info(line)
        input("press a key to continue")
