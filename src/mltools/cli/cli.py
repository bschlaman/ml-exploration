import argparse
import logging
import random
import textwrap
import sys

import mltools.modules.k_nearest_neighbors.knn
import mltools.modules.law_large_numbers.lln
import mltools.modules.linear_classifiers
import mltools.modules.perceptron.perceptron
import mltools.modules.transformers
from mltools.constants import RANDOM_SEED
from mltools.utils.formatting.colors import bld, blu, mag

logging.basicConfig(
    format="[%(levelname)-8s] (%(name)s) %(message)s",
    level=logging.DEBUG,
)

log = logging.getLogger(__name__)

APPLICATION_NAME = "ML Exploration Modules"
# TODO: can I get this dynamically?
BIN_NAME = "mltools"


def main():
    # making a parse of myself
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            f"""{bld(APPLICATION_NAME)}
            This tool set provides utilities, demos, and POCs for
            various machine learning concepts.  Below are a list of modes.
            """
        ),
    )
    modules_arg_group = parser.add_mutually_exclusive_group(required=True)
    modules_arg_group.add_argument(
        "--lln",
        action="store_true",
        help="Demo of expected value convergence caused by the Law of Large Numbers",
    )
    modules_arg_group.add_argument(
        "--knn",
        action="store_true",
        help="Demo of the curse of dimensionality in the KNN algorithm",
    )
    modules_arg_group.add_argument(
        "--perceptron",
        action="store_true",
        help="Matplotlib demo of the Perceptron",
    )
    modules_arg_group.add_argument(
        "--linear-classifiers",
        action="store_true",
        help="Comparison of linear classifiers",
    )
    modules_arg_group.add_argument(
        "--transformers",
        action="store_true",
        help="Demo of transformers as utilized in LLMs",
    )
    args = parser.parse_args()

    log.info(mag(APPLICATION_NAME))
    log.info(f"{blu('try running ')}{BIN_NAME} -h{blu(' for a list of modes')}")

    if args.lln:
        mltools.modules.law_large_numbers.lln.lln_convergence()

    if args.knn:
        mltools.modules.k_nearest_neighbors.knn.curse_of_dimensionality()

    if args.perceptron:
        mltools.modules.perceptron.perceptron.perceptron2D()

    if args.linear_classifiers:
        mltools.modules.linear_classifiers.entry.run_classifiers()

    if args.transformers:
        mltools.modules.transformers.entry.run()

    log.info(blu("Welcome to the ml module"))
    random.seed(RANDOM_SEED)
    log.debug(f"using random seed: {RANDOM_SEED}")
