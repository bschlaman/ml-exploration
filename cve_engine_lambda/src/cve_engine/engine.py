from functools import wraps
import pickle
import numpy as np
import os
import logging
from datetime import datetime

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.feature_extraction.text import CountVectorizer

from bpyutils.formatting.colors import yel, red, mag, bld

from cve_engine.cvss_data import CVSS_BASE_METRICS

log = logging.getLogger(__name__)


class CVEEngineModel:
    def __init__(self):
        # assume same learn rate for each metric
        self.learn_rate = 0.04
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.training_epochs = 2000

        self.models = {}
        self.optimizers = {}

        self.model_save_dir = "../data/models/cve_engine"

    @staticmethod
    def _ensure_initialized(func):
        @wraps(func)
        def _wrap(self, *args, **kwargs):
            # TODO (2023.08.06) how should I deal with the optimizers?
            # When loading the models, the optimizers will be blank
            required = [
                "bow_vec",
                "models",
                # "optimizers",
            ]
            for attr in required:
                if not hasattr(self, attr) or not getattr(self, attr):
                    log.error(f"missing: {attr}")
                    raise ValueError(
                        f"{self.__class__.__name__} must be initialized; call new_model or load"
                    )
            return func(self, *args, **kwargs)

        return _wrap

    def new_model(self, bow_vec: CountVectorizer):
        """Use if not loading previous models"""

        self.bow_vec = bow_vec
        self.writer = SummaryWriter(log_dir="../data/logs/torch")

        for metric_meta in CVSS_BASE_METRICS.values():
            model = torch.nn.Linear(
                len(bow_vec.vocabulary_), len(metric_meta.categories)
            )
            self.models[metric_meta.abbrev] = model
            self.optimizers[metric_meta.abbrev] = torch.optim.SGD(
                model.parameters(), lr=self.learn_rate
            )
        assert self.models.keys() == self.optimizers.keys()

    @_ensure_initialized
    def display_parameters(self):
        print(bld("Model details"))
        print("-------------")
        for metric, model in self.models.items():
            print(
                f"metric {yel(metric).rjust(len(yel('')) + 2)} categories:"
                f" {red(model.out_features)}"
            )
        print()
        print(bld("Other parameters"))
        print("----------------")
        print(f"learn rate:             {mag(str(self.learn_rate))}")
        print(
            f"feature dimensionality: {mag(list(self.models.values())[0].in_features)}"
        )
        print(f"epochs per metric:      {mag(str(self.training_epochs))}")
        print()

    @_ensure_initialized
    def _validate_Y_properties(self, Y_train: torch.Tensor):
        assert Y_train.shape[1] == len(self.models)
        values, _ = torch.max(Y_train, dim=0)
        assert list(values) == [
            len(metric_meta.categories) - 1
            for metric_meta in CVSS_BASE_METRICS.values()
        ]

    @_ensure_initialized
    def _train_metric(self, X_train: torch.Tensor, Y_train: torch.Tensor, metric: str):
        for epoch in range(self.training_epochs):
            self.optimizers[metric].zero_grad()
            outputs = self.models[metric](X_train)

            loss = self.loss_fn(outputs, Y_train)
            loss.backward()

            self.writer.add_scalar(f"Loss/train/{metric}", loss, epoch)

            self.optimizers[metric].step()

            if epoch % 100:
                continue
            log.debug(f"metric: {metric:2}\tepoch: {epoch:3}\tloss: {loss}")

    @_ensure_initialized
    def preprocess_examples(self, X_np: np.ndarray) -> torch.Tensor:
        """Vectorize examples and convert to torch.Tensor"""
        log.debug("transforming training examples...")
        vectorized = self.bow_vec.transform(X_np).toarray()
        return torch.from_numpy(vectorized).float()

    @_ensure_initialized
    def train_all(self, X_train_np: np.ndarray, Y_train: torch.Tensor):
        """Trains all models on the provided training data.
        :param metric_labels: a map from CVSS metric metriciations
                              to the Y_train torch.Tensor that is
                              associated with X_train.
        """
        self._validate_Y_properties(Y_train)
        X_train = self.preprocess_examples(X_train_np)

        for i, metric in enumerate(self.models.keys()):
            log.debug(f"++ training metric {i}: {metric}")
            self._train_metric(X_train, Y_train[:, i], metric)

        log.info("training complete; flushing writer")
        self.writer.flush()

    @_ensure_initialized
    def predict(self, X_test_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns predictions and confidence scores
        indexed by cvss metric"""

        X = self.preprocess_examples(X_test_np)

        predictions = np.zeros((X.shape[0], len(self.models)))
        confidence_scores = np.zeros((X.shape[0], len(self.models)))

        for i, (metric, model) in enumerate(self.models.items()):
            log.debug(f"predict: {metric}")
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

    def _create_dir_path(self):
        dir_path = os.path.join(
            self.model_save_dir, datetime.now().strftime("%Y.%m.%d-%s")
        )
        log.info(f"saving models to {dir_path}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path

    def locate_latest_models_path(self):
        return max(os.listdir(self.model_save_dir), key=lambda _: tuple(_.split("-")))

    def load_latest_models(self):
        """Convenience method for loading the latest models
        and vectorizer as determined by their saved location"""
        latest_path = self.locate_latest_models_path()
        log.debug(f"loading latest models from {latest_path}")
        self.load_models_full(latest_path)

    @_ensure_initialized
    def save_models_full(self):
        dir_path = self._create_dir_path()
        for metric, model in self.models.items():
            torch.save(
                model,
                os.path.join(
                    dir_path,
                    f"{metric}_model.pth",
                ),
            )
        with open(
            os.path.join(self.model_save_dir, models_root_dir, "bow_vec.pkl"), "wb"
        ) as f:
            pickle.dump(self.bow_vec, f)

    def load_models_full(self, models_root_dir: str):
        """Load the models into self.models and vectorizer into self.bow_vec,
        overwriting whatever was stored there."""
        for metric in CVSS_BASE_METRICS:
            self.models[metric] = torch.load(
                os.path.join(
                    self.model_save_dir, models_root_dir, f"{metric}_model.pth"
                )
            )
        with open(
            os.path.join(self.model_save_dir, models_root_dir, "bow_vec.pkl"), "rb"
        ) as f:
            self.bow_vec = pickle.load(f)
        assert (
            self.models.keys() == CVSS_BASE_METRICS.keys()
        ), "should be no leftover metric keys"
        assert len(self.bow_vec.vocabulary_) > 0, "bow_vec should be valid"
