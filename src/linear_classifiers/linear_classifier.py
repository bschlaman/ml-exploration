from abc import ABC, abstractmethod
import collections
import random
import itertools
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
        words = list(map(lambda w: "".join(filter(str.isalnum, w)), x.lower().split()))
        return (words, bool(int(y)))

    return map(_conv_xy, ((rd[features_key], rd[label_key]) for rd in raw_data))


class LinearClassifier(ABC):
    w: Vector

    def __init__(self, data: tuple[list[str], bool]):
        self.data = []
        self.m = 6  # number of words per datum
        for features, label in data:
            self.data.append((self._sample_features(features), label))

    def _sample_features(self, features: list[str]) -> collections.Counter:
        """Convert raw data into feature vectors that can be operated on"""
        # not sure if this is the same as the dimensionality
        # since there will likely be >>6 distinct words across data points
        return collections.Counter(random.choices(features, k=self.m))

    # core components of a classifier
    @abstractmethod
    def calculate_parameter(self, c: int, alpha: str) -> float:
        """Estimate θ^_αc for a generative algorithm with
        multinomial features
        """
        pass

    def get_unique_words(self) -> set[str]:
        return set(itertools.chain(*(_[0].keys() for _ in self.data)))

    def log_data_stats(self):
        num_unique_words = len(self.get_unique_words())
        label_counts = collections.Counter((_[1] for _ in self.data))
        labeled_data = {
            "num datapoints": len(self.data),
            "label counts": label_counts,
            "num unique words": num_unique_words,
            "P(y = true)": label_counts[True] / len(self.data),
            "P(y = false)": label_counts[False] / len(self.data),
        }
        for line in data_print(labeled_data, Fore.YELLOW):
            log.info(line)


class NaiveBayesClassifier(LinearClassifier):
    def __init__(self):
        raw_data = list(data_iter())
        super().__init__(list(transform_raw_data(raw_data)))

    def calculate_parameter(self, c: int, alpha: str) -> float:
        """Estimate θ^_αc for a generative algorithm with
        multinomial features
        """
        # TODO (2022.11.20) add smoothing; should I assume I've seen each word once?
        # sum(I(y_i == c)x_iα); i.e. number of times word α
        # has appeared across all data points
        alpha_count = 0
        # TODO (2022.11.20) is this the correct denominator?
        word_count = self.m * len(self.data)
        for features, label in self.data:
            if label != c:
                continue
            alpha_count += features[alpha]
        return alpha_count / word_count
