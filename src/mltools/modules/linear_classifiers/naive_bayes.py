import collections
import logging
import random

from bpyutils.formatting.colors import mag

from mltools.modules.linear_classifiers.multinomial_linear_classifier import \
    MultinomialLinearClassifier

log = logging.getLogger(__name__)


class MultinomialNaiveBayes(MultinomialLinearClassifier):
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
        labels = self._get_unique_labels()
        self.model = {y: collections.defaultdict(float) for y in labels}
        for y in labels:
            log.info(f"calculating params for label: {mag(str(y))}")
            # TODO (2022.12.02): sorting the set to make order deterministic.
            # is there a better way?
            for i, word in enumerate(sorted(self.dictionary)):
                param = self.calculate_parameter(y, word, plus_one_smoothing=True)
                if i % 502 == 0:
                    log.debug(f"{word=:>20}: {param:0.5}")
                self.model[y][word] = param

    def _test_datapoint(self, datapoint: dict[str, int], c: bool) -> float:
        # start with the label weights, π_c
        prod = self.label_weights[c]
        for word, count in datapoint.items():
            prod *= pow(self.model[c][word], count)
        return prod

    def predict(self):
        correct = 0
        for words, label in self.test_data:
            outcome = {}
            for y in self._get_unique_labels():
                outcome[y] = self._test_datapoint(words, y)

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
