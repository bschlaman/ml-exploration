from abc import ABC, abstractmethod
import collections
import random

from utils.math.vectors import Vector
from linear_classifiers.data_generator import data_iter, get_data_stats


def transform_raw_data(raw_data: list[dict[str, str]]) -> tuple[list[str], bool]:
    """Convert raw data into the format the classifiers expect"""
    label_key = "real"
    features_key = "title"
    data = []
    for rd in raw_data:
        data.append((rd[features_key].split(), bool(rd[label_key])))
    return data


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


class NaiveBayesClassifier(LinearClassifier):
    def __init__(self):
        get_data_stats()
        raw_data = list(data_iter())[:10]
        super().__init__(transform_raw_data(raw_data))

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
