import enum
import logging
import random

from constants import RANDOM_SEED
from utils.printing import blu

logging.basicConfig(
    format="[%(levelname)-8s] (%(name)s) %(message)s",
    level=logging.DEBUG,
)

log = logging.getLogger(__name__)


class DemoMode(enum.Enum):
    LAW_LARGE_NUMBERS = "LAW_LARGE_NUMBERS"
    K_NEAREST_NEIGHBORS = "K_NEAREST_NEIGHBORS"
    PERCEPTRON = "PERCEPTRON"
    LINEAR_CLASSIFIERS = "LINEAR_CLASSIFIERS"
    TRANSFORMERS = "TRANSFORMERS"


DEMO_MODE = DemoMode.TRANSFORMERS


def main():
    log.info(blu("Welcome to the ml module"))
    random.seed(RANDOM_SEED)
    log.debug(f"using random seed: {RANDOM_SEED}")

    if DEMO_MODE == DemoMode.LAW_LARGE_NUMBERS:
        from law_large_numbers import lln

        log.info(f"starting expected value convergence test...")
        lln.lln_convergence()

    if DEMO_MODE == DemoMode.K_NEAREST_NEIGHBORS:
        from k_nearest_neighbors import knn

        log.info(f"starting curse of dimensionality test...")
        knn.curse_of_dimensionality()

    if DEMO_MODE == DemoMode.PERCEPTRON:
        from perceptron import perceptron

        log.info(f"starting perceptron demo...")
        perceptron.perceptron2D()

    if DEMO_MODE == DemoMode.LINEAR_CLASSIFIERS:
        import linear_classifiers.entry

        log.info(f"starting linear classifier demo with multinomial inputs...")
        linear_classifiers.entry.run_classifiers()

    if DEMO_MODE == DemoMode.TRANSFORMERS:
        import transformers.entry

        log.info("starting transformer demo...")
        transformers.entry.run()


if __name__ == "__main__":
    main()
