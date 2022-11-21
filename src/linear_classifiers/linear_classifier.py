from abc import ABC, abstractmethod
from utils.math.vectors import Vector
from linear_classifiers.data_generator import data_gen


class LinearClassifier(ABC):
    w: Vector

    def __init__(self, data: list[Vector], labels: list):
        assert len(data) == len(labels)
        self.data = data
        self.labels = labels


class NaiveBayesClassifier(LinearClassifier):
    def __init__(self):
        next(data_gen())
        super().__init__([Vector(1, 2, 3)], [123])
