import logging
import random
import heapq
from colorama import Fore
from constants import RANDOM_SEED
from law_large_numbers import lln
from k_nearest_neighbors import knn
from perceptron import perceptron
from linear_classifiers.linear_classifier import NaiveBayesClassifier

logging.basicConfig(
    # format="%(asctime)s [%(levelname)-8s] (%(name)s) %(message)s",
    format="[%(levelname)-8s] (%(name)s) %(message)s",
    level=logging.DEBUG,
)

log = logging.getLogger(__name__)


def main():
    log.info(f"{Fore.BLUE}Welcome to the ml module{Fore.RESET}")
    random.seed(RANDOM_SEED)
    log.debug(f"using random seed: {RANDOM_SEED}")

    # log.info(f"starting expected value convergence test...")
    # lln.lln_convergence()
    # log.info(f"starting curse of dimensionality test...")
    # knn.curse_of_dimensionality()
    # log.info(f"starting perceptron demo...")
    # log.info(f"starting perceptron test...")
    # perceptron.perceptron2D()

    log.info("starting linear classifier test...")
    nbc = NaiveBayesClassifier()
    nbc.log_data_stats()

    heap = []

    label_probability_false = 0
    unique_words = nbc.get_unique_words()
    for i, word in enumerate(unique_words):
        param = nbc.calculate_parameter(False, word)
        label_probability_false += param
        if i % 502 == 0:
            log.info(f"{word=:>20}: {param}")

        if len(heap) < 25:
            heapq.heappush(heap, (param, word))
        elif param > heap[0][0]:
            heapq.heapreplace(heap, (param, word))

    log.info(f"{label_probability_false=}")
    log.info("top 25 most likely words for fake news")
    log.info("======================================")
    for param, word in sorted(heap, reverse=True):
        log.info(f"{word=:>20}: {param}")


if __name__ == "__main__":
    main()
