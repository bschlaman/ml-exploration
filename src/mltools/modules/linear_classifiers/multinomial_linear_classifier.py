import collections
import heapq
import itertools
import logging
from abc import ABC, abstractmethod

from mltools.modules.linear_classifiers.data_generator import data_iter
from mltools.utils.formatting import std
from mltools.utils.math.vectors import Vector

log = logging.getLogger(__name__)


class MultinomialLinearClassifier(ABC):
    w: Vector
    model: dict[bool, dict[str, float]]

    def __init__(self):
        self.data: list[tuple[dict, bool]] = list(data_iter())

        split = int(len(self.data) * 0.7)
        self.training_data = self.data[:split]
        self.test_data = self.data[split:]

        # upfront calculations
        self.label_counts = collections.Counter((_[1] for _ in self.training_data))
        self.label_weights = self._get_label_weights()
        self.num_words_by_label = self._get_num_words_by_label()
        self.dictionary = self._get_unique_words()

    def _get_unique_words(self) -> set[str]:
        return set(itertools.chain(*(_[0].keys() for _ in self.training_data)))

    def _get_unique_labels(self) -> set[bool]:
        """Returns a set of possible labels by iterating
        over the training dataset
        """
        return set(_[1] for _ in self.training_data)

    def _get_label_weights(self) -> dict[bool, float]:
        """P(y)"""
        return {
            y: count / len(self.training_data) for y, count in self.label_counts.items()
        }

    def _get_num_words_by_label(self):
        return {
            y: sum(
                sum(features.values())
                for features, label in self.training_data
                if y == label
            )
            for y in self._get_unique_labels()
        }

    def log_data_stats(self):
        num_unique_words = len(self.dictionary)
        labeled_data = {
            "num datapoints": len(self.training_data),
            "label counts": self.label_counts,
            "num unique words": num_unique_words,
            "P(y = true)": self.label_weights[True],
            "P(y = false)": self.label_weights[False],
        }
        for line in std.data_print(labeled_data):
            log.info(line)

    def get_top_n_params(self, c: bool, n: int) -> list[tuple[float, str]]:
        # TODO: lol just use n_largest
        heap = []
        for word in self.dictionary:
            param = self.model[c][word]
            if len(heap) < n:
                heapq.heappush(heap, (param, word))
            elif param > heap[0][0]:
                heapq.heapreplace(heap, (param, word))
        return heap

    def constant_classifier(self) -> bool:
        """Error upper bound.  In classification, this is the most common label.
        In regression settings, the constant sould minimize loss on the training set.
        This value represents a baseline that any decent classifier should beat
        with statistical significance.
        """
        return max(self.label_counts, key=self.label_counts.get)

    # core components of a classifier
    # need to revisit this component; does it have an analog for knn?
    @abstractmethod
    def calculate_parameter(self, c: bool, alpha: str) -> float:
        """Estimate θ^_αc for a generative algorithm with
        multinomial features
        """
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
