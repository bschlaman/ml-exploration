from abc import ABC, abstractmethod
import collections
import random
import itertools
import heapq
import logging
from colorama import Fore

from utils.math.vectors import Vector
from utils.helpers import data_print
from linear_classifiers.data_generator import data_iter


log = logging.getLogger(__name__)


def transform_raw_data(raw_data: list[dict[str, str]]) -> tuple[list[str], bool]:
    """Convert raw data into the format the classifiers expect"""
    label_key = "real"
    features_key = "title"

    def _conv_xy(d):
        x, y = d
        words = list(
            # word after non alnum char extraction must not be empty
            filter(
                bool, map(lambda w: "".join(filter(str.isalnum, w)), x.lower().split())
            )
        )
        return (words, bool(int(y)))

    return map(_conv_xy, ((rd[features_key], rd[label_key]) for rd in raw_data))


class LinearClassifier(ABC):
    w: Vector

    def __init__(self, data: list[tuple[list[str], bool]]):
        self.data = []
        self.m = 6  # number of words per datum
        for features, label in data:
            self.data.append((self._sample_features(features), label))

        split = int(len(self.data) * 0.7)
        self.training_data = self.data[:split]
        self.test_data = self.data[split:]

        # upfront calculations
        self.label_counts = collections.Counter((_[1] for _ in self.training_data))
        self.label_weights = self._get_label_weights()

    def _sample_features(self, features: list[str]) -> collections.Counter:
        """Convert raw data into feature vectors that can be operated on"""
        # not sure if this is the same as the dimensionality
        # since there will likely be >>6 distinct words across data points
        return collections.Counter(random.choices(features, k=self.m))

    # core components of a classifier
    @abstractmethod
    def calculate_parameter(self, c: bool, alpha: str) -> float:
        """Estimate θ^_αc for a generative algorithm with
        multinomial features
        """
        pass

    def get_unique_words(self) -> set[str]:
        return set(itertools.chain(*(_[0].keys() for _ in self.training_data)))

    def get_unique_labels(self) -> set[bool]:
        return set(_[1] for _ in self.training_data)

    def _get_label_weights(self) -> dict[bool, float]:
        """P(y)"""
        return {
            l: self.label_counts[l] / len(self.training_data)
            for l in self.get_unique_labels()
        }

    def log_data_stats(self):
        num_unique_words = len(self.get_unique_words())
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
        heap = []
        for word in self.get_unique_words():
            param = self.model[c][word]
            if len(heap) < n:
                heapq.heappush(heap, (param, word))
            elif param > heap[0][0]:
                heapq.heapreplace(heap, (param, word))
        return heap

    @abstractmethod
    def train(self):
        pass


class NaiveBayesClassifier(LinearClassifier):
    def __init__(self):
        raw_data = list(data_iter())
        super().__init__(list(transform_raw_data(raw_data)))

    def calculate_parameter(self, c: bool, alpha: str) -> float:
        """Estimate θ^_αc for a generative algorithm with
        multinomial features
        """
        # TODO (2022.11.20) add smoothing; should I assume I've seen each word once?
        # sum(I(y_i == c)x_iα); i.e. number of times word α
        # has appeared across all data points
        alpha_count = 0
        num_words_for_label = self.m * self.label_counts[c]
        for features, label in self.training_data:
            if label != c:
                continue
            alpha_count += features[alpha]
        return alpha_count / num_words_for_label

    # TODO (2022.12.01): can this be algo agnostic?
    def train(self):
        """Note: this function does not promise to be perfectly optimized!!!
        This is a learning tool that prefers helpful text output over performance
        """
        labels = self.get_unique_labels()
        self.model = {l: collections.defaultdict(float) for l in labels}
        for l in labels:
            log.info(f"calculating params for label: {l}")
            # TODO (2022.12.02): sorting the set to make order deterministic.
            # is there a better way?
            for i, word in enumerate(sorted(self.get_unique_words())):
                param = self.calculate_parameter(l, word)
                if i % 502 == 0:
                    log.info(f"{word=:>20}: {param:0.5}")
                self.model[l][word] = param

    def test_datapoint(self, datapoint: collections.Counter, c: bool):
        # start with the weights, π_c
        prod = self.label_weights[c]
        for word, count in datapoint.items():
            prod *= pow(self.model[c][word], count)
        return prod

    def predict(self):
        correct = 0
        for words, label in self.test_data:
            outcome = {}
            for l in self.get_unique_labels():
                outcome[l] = self.test_datapoint(words, l)
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
    def __init__(self):
        raw_data = list(data_iter())
        super().__init__(list(transform_raw_data(raw_data)))
