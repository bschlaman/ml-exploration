import collections
import math
import random
import logging
import heapq

from linear_classifiers.multinomial_linear_classifier import MultinomialLinearClassifier

log = logging.getLogger(__name__)


def euclidian_distance(p1: dict[str, int], p2: dict[str, int]) -> float:
    total = 0
    for word in p1.keys() | p2.keys():
        total += abs(p1.get(word, 0) - p2.get(word, 0)) ** 2
    return math.sqrt(total)


class MultinomialKNearestNeighbors(MultinomialLinearClassifier):
    def __init__(self, k: int):
        if k % 2 == 0:
            raise Exception("use odd k so that there are no classification ties")
        self.k = k
        super().__init__()

    # TODO (2023.03.13): can this be removed?
    def calculate_parameter(self, c: bool, alpha: str) -> float:
        return super().calculate_parameter(c, alpha)

    def _test_datapoint(self, datapoint: dict[str, int]) -> bool:
        heap = []  # the distances
        counter = collections.defaultdict(int)
        for point, label in self.training_data:
            distance = euclidian_distance(datapoint, point)
            if len(heap) < self.k:
                heapq.heappush(heap, (-distance, label))
                counter[label] += 1
            elif distance < -heap[0][0]:
                _, l = heapq.heapreplace(heap, (-distance, label))
                counter[label] += 1
                counter[l] -= 1

        return max(counter, key=counter.get)

    def train(self):
        """KNN is a lazy-learning algorithm; training is a no-op"""
        return

    def predict(self):
        correct = 0
        for i, (words, label) in enumerate(self.test_data):
            log.debug(f"{i}/{len(self.test_data)} testing point: {words}")
            prediction = self._test_datapoint(words)

            if prediction == label:
                correct += 1

            if random.random() < 0.01:
                log.debug(f"testing: {words}")
                log.debug(f"expect:  {label}")
                log.debug(f"got:     {prediction}")
        print(correct, len(self.test_data), correct / len(self.test_data))
