from abc import ABC, abstractmethod
from utils.math.vectors import Vector
from linear_classifiers.data_generator import data_iter, get_data_stats


class LinearClassifier(ABC):
    w: Vector

    def __init__(self, data: list[Vector], labels: list):
        assert len(data) == len(labels)
        self.data = data
        self.labels = labels

    def _build_features(words: list[str]) -> collections.Counter:
        """Convert raw data into feature vectors than can be operated on
        This must be the same process for all classifiers
        """
        # not sure if this is the same as the dimensionality
        # since there will likely be >>6 distinct words across data points
        num_words = 6
        return collections.Counter(random.choices(words, k=num_words))


class NaiveBayesClassifier(LinearClassifier):
    def __init__(self):
        get_data_stats()
        super().__init__([Vector(1, 2, 3)], [123])
        # generative learning

    def _build_features():
        pass
