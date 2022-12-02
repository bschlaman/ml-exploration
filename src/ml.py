import logging
import random
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
    nbc.train()

    label_probability_false = sum(nbc.model[False].values())
    log.info(f"{label_probability_false=}")

    label_probability_true = sum(nbc.model[True].values())
    log.info(f"{label_probability_true=}")

    for label in nbc.get_unique_labels():
        log.info(f"top 25 most likely words for {label} news")
        log.info("======================================")
        for param, word in sorted(nbc.get_top_n_params(label, 24), reverse=True):
            log.info(f"{word=:>20}: {param}")


if __name__ == "__main__":
    main()
