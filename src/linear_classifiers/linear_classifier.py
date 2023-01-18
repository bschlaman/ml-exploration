from abc import ABC, abstractmethod
import random
import collections
import itertools
import heapq
import logging
from colorama import Fore

from utils.math.vectors import Vector
from utils.helpers import data_print
from linear_classifiers.data_generator import data_iter


log = logging.getLogger(__name__)


class LinearClassifier(ABC):
    w: Vector

    def __init__(self):
        self.data: list[tuple[list[str], bool]] = list(data_iter())

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

    def get_unique_labels(self) -> set[bool]:
        return set(_[1] for _ in self.training_data)

    def _get_label_weights(self) -> dict[bool, float]:
        """P(y)"""
        return {
            y: self.label_counts[y] / len(self.training_data)
            for y in self.get_unique_labels()
        }

    def _get_num_words_by_label(self):
        return {
            y: sum(
                sum(features.values())
                for features, label in self.training_data
                if y == label
            )
            for y in self.get_unique_labels()
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
        for line in data_print(labeled_data, Fore.YELLOW):
            log.info(line)

    def get_top_n_params(self, c: bool, n: int) -> tuple[float, str]:
        # TODO: lol just use n_largest
        heap = []
        for word in self.dictionary:
            param = self.model[c][word]
            if len(heap) < n:
                heapq.heappush(heap, (param, word))
            elif param > heap[0][0]:
                heapq.heapreplace(heap, (param, word))
        return heap

    def constant_classifier(self):
        """Error upper bound.  In classification, this is the most common label.
        In regression settings, the constant sould minimize loss on the training set.
        This value represents a baseline that any decent classifier should beat
        with statistical significance.
        """
        return max(self.label_weights.values())

    # core components of a classifier
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


class NaiveBayesClassifier(LinearClassifier):
    def calculate_parameter(
        self, c: bool, alpha: str, plus_one_smoothing: bool = False
    ) -> float:
        """Estimate θ^_αc for a generative algorithm with
        multinomial features
        """
        l = int(plus_one_smoothing)
        # sum(I(y_i == c)x_iα); i.e. number of times word α
        # has appeared across all data points with y == c
        alpha_count = l
        for features, label in self.training_data:
            if label != c:
                continue
            alpha_count += features[alpha]
        return alpha_count / (self.num_words_by_label[c] + l * len(self.dictionary))

    # TODO (2022.12.01): can this be algo agnostic?
    def train(self):
        """Note: this function does not promise to be perfectly optimized!!!
        This is a learning tool that prefers helpful text output over performance.
        """
        labels = self.get_unique_labels()
        self.model = {y: collections.defaultdict(float) for y in labels}
        for y in labels:
            log.info(f"calculating params for label: {y}")
            # TODO (2022.12.02): sorting the set to make order deterministic.
            # is there a better way?
            for i, word in enumerate(sorted(self.dictionary)):
                param = self.calculate_parameter(y, word, plus_one_smoothing=True)
                if i % 502 == 0:
                    log.info(f"{word=:>20}: {param:0.5}")
                self.model[y][word] = param

    def test_datapoint(self, datapoint: collections.Counter, c: bool):
        # start with the label weights, π_c
        prod = self.label_weights[c]
        for word, count in datapoint.items():
            prod *= pow(self.model[c][word], count)
        return prod

    def predict(self):
        correct = 0
        for words, label in self.test_data:
            outcome = {}
            for y in self.get_unique_labels():
                outcome[y] = self.test_datapoint(words, y)

            if len(set(outcome.values())) == 1:
                res = self.constant_classifier()
            else:
                res = max(outcome, key=outcome.get)

            if label == res:
                correct += 1

            if random.random() < 0.001:
                log.debug(f"testing: {words}")
                log.debug(f"expect:  {label}")
                log.debug(f"got:     {res}")
                print(outcome)
        print(correct, len(self.test_data), correct / len(self.test_data))


class KNearestNeighbors(LinearClassifier):
    def __init__(self, k: int):
        self.k = k
        super().__init__()

    def test_datapoint(self, datapoint: collections.Counter, c: bool):
        pass
