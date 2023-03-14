import collections

from linear_classifiers.multinomial_linear_classifier import MultinomialLinearClassifier


class MultinomialKNearestNeighbors(MultinomialLinearClassifier):
    def __init__(self, k: int):
        self.k = k
        super().__init__()

    def test_datapoint(self, datapoint: collections.Counter, c: bool):
        pass
