import logging

from bpyutils.formatting.colors import grn

from mltools.modules.linear_classifiers.knn import MultinomialKNearestNeighbors
from mltools.modules.linear_classifiers.naive_bayes import \
    MultinomialNaiveBayes

log = logging.getLogger(__name__)


def run_classifiers():
    log.info(f"testing classifier: {grn(MultinomialNaiveBayes.__name__)}")
    nbc = MultinomialNaiveBayes()
    nbc.log_data_stats()
    nbc.train()

    label_probability_false = sum(nbc.model[False].values())
    log.info(f"{label_probability_false=}")

    label_probability_true = sum(nbc.model[True].values())
    log.info(f"{label_probability_true=}")

    print(nbc._get_label_weights())

    n = 10
    for label in nbc._get_unique_labels():
        log.info(f"top {n} most likely words for {label} news")
        log.info("======================================")
        for param, word in sorted(nbc.get_top_n_params(label, n), reverse=True):
            log.info(f"{word=:>20}: {param:0.5}")

    nbc.predict()

    log.info(f"testing classifier: {grn(MultinomialKNearestNeighbors.__name__)}")
    knnc = MultinomialKNearestNeighbors(111)
    knnc.predict()
