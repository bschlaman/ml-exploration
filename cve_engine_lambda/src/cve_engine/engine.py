import torch
import numpy as np
import os
import nltk
import string
import logging

from cve_engine.cvss import CVSS_BASE_METRICS

log = logging.getLogger(__name__)


def desc_preprocess(d: str):
    log.debug("preprocessing description...")
    # setup
    stopwords = set(nltk.corpus.stopwords.words("english"))
    lemmatizer = nltk.stem.WordNetLemmatizer()

    # lowercase
    d = d.lower()
    # remove punctuation
    d = d.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    # tokenize
    tokens = d.split()
    # remove stop words
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


class CVEEngineModel:
    def __init__(self):
        self.models_asset_path = "assets/models/cve_engine"
        self.models: dict[str, torch.nn.modules.linear.Linear] = {}
        latest_models_path = self.locate_latest_models_path()
        log.debug(f"loading latest models from: {latest_models_path}")
        self.load_metric_models(latest_models_path)

    def load_metric_models(self, models_root_dir: str):
        for metric in CVSS_BASE_METRICS:
            model = torch.load(
                os.path.join(
                    self.models_asset_path, models_root_dir, f"{metric}_model.pth"
                ),
                map_location=torch.device("cpu"),
            )
            model.eval()
            self.models[metric] = model

    def locate_latest_models_path(self):
        return max(os.listdir(self.models_asset_path), key=lambda _: tuple(_.split("-")))

    def display_parameters(self):
        print("== models ==")
        for metric, model in self.models.items():
            print(f"metric: {metric}\tcategories: {model.out_features}")

        print("\n== parameters ==")
        print(f"feature dimensionality: {list(self.models.values())[0].in_features}")

    def _validate_Y_properties(self, Y_train: torch.Tensor):
        assert Y_train.shape[1] == len(self.models)
        values, _ = torch.max(Y_train, dim=0)
        assert list(values) == [
            len(metric_meta.categories) - 1
            for metric_meta in CVSS_BASE_METRICS.values()
        ]

    def predict(self, X: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Returns predictions and confidence scores
        indexed by cvss metric"""

        predictions = np.zeros((X.shape[0], len(self.models)))
        confidence_scores = np.zeros((X.shape[0], len(self.models)))

        for i, (metric, model) in enumerate(self.models.items()):
            prob = torch.nn.functional.softmax(model(X), dim=1)

            pred = torch.argmax(prob, dim=1)

            confidence = prob[range(prob.shape[0]), pred]

            predictions[:, i] = pred.numpy()
            confidence_scores[:, i] = confidence.detach().numpy()

        assert predictions.shape == confidence_scores.shape
        return predictions, confidence_scores

    @staticmethod
    def compute_accuracy(Y_true: np.ndarray, Y_pred: np.ndarray):
        """Computes the accuracy for each metric
        by measuring the proportion of correct predictions"""
        assert Y_true.shape == Y_pred.shape
        return np.mean(Y_true == Y_pred, axis=0)
