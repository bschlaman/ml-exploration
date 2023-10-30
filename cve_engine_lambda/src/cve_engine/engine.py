from functools import wraps
import pickle
import numpy as np
import os
import logging
from datetime import datetime
import time

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer

from bpyutils.formatting.colors import yel, red, mag, bld

from cve_engine.cvss_data import CVSS_BASE_METRICS

log = logging.getLogger(__name__)


class CVEEngineModel:
    """A stateful container for methods that create,
    save, load, and train CVEEngine models.
    """

    def __init__(self):
        # assume same learn rate for each metric
        self.learn_rate = 0.15
        self.criterion = torch.nn.CrossEntropyLoss()
        self.training_epochs = 4000

        self.models = {}
        self.optimizers = {}

        # TODO (2023.10.28): the path must have preceding ".."
        # if using this in a notebook; find a long term solution
        # that works for all environments.
        self.model_save_dir = "data/models/cve_engine"
        self.ipex_optimized = False
        # toggle this parameter to switch off cuda
        self.cuda_enabled = torch.cuda.is_available()

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
                        f"{self.__class__.__name__} must be initialized;"
                        " call new_model or load"
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
        print(f"ipex optimized:         {bld(str(self.ipex_optimized))}")
        print(f"cuda available:         {bld(str(torch.cuda.is_available()))}")
        print(f"cuda enabled:           {bld(str(self.cuda_enabled))}")
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
        start_time = time.perf_counter()
        for epoch in range(self.training_epochs):
            self.optimizers[metric].zero_grad()
            outputs = self.models[metric](X_train)

            loss = self.criterion(outputs, Y_train)
            loss.backward()

            self.writer.add_scalar(f"Loss/train/{metric}", loss, epoch)

            self.optimizers[metric].step()

            if epoch % 100:
                continue
            time_elapsed = time.perf_counter() - start_time
            log.debug(
                f"metric: {metric:2}\tepoch: {epoch:4}\tloss: {loss:0.5}\telapsed: {time_elapsed:0.4}"
            )
        log.debug(
            f"total training time metric {metric:2}: {time.perf_counter() - start_time:0.5}"
        )

    @_ensure_initialized
    def _train_metric_v2(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_test: torch.Tensor,
        Y_test: torch.Tensor,
        metric: str,
    ):
        """This version of `_train_metric` also tracks validation metric"""
        start_time = time.perf_counter()

        train_loader = DataLoader(
            TensorDataset(X_train, Y_train), batch_size=32, shuffle=True
        )
        val_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=32)

        for epoch in range(self.training_epochs):
            self.models[metric].train()
            train_losses = []
            for X_batch, Y_batch in train_loader:
                self.optimizers[metric].zero_grad()
                outputs = self.models[metric](X_batch)
                loss = self.criterion(outputs, Y_batch)
                loss.backward()
                self.optimizers[metric].step()
                train_losses.append(loss.item())

            avg_train_loss = sum(train_losses) / len(train_losses)
            self.writer.add_scalar(f"Loss/train/{metric}", avg_train_loss, epoch)

            # Validation
            self.models[metric].eval()
            with torch.no_grad():
                val_losses = [
                    self.criterion(self.models[metric](X), Y).item()
                    for X, Y in val_loader
                ]

            avg_val_loss = sum(val_losses) / len(val_losses)
            self.writer.add_scalar(f"Loss/val/{metric}", avg_val_loss, epoch)

            if epoch % 100:
                continue
            time_elapsed = time.perf_counter() - start_time
            log.debug(
                f"metric: {metric:2}\tepoch: {epoch:4}\tloss: {avg_train_loss:0.5}\telapsed: {time_elapsed:0.4}"
            )
        log.debug(
            f"total training time metric {metric:2}: {time.perf_counter() - start_time:0.5}"
        )

    @_ensure_initialized
    def preprocess_examples(self, X_np: np.ndarray) -> torch.Tensor:
        """Vectorize examples and convert to torch.Tensor"""
        log.debug("transforming training examples...")
        vectorized = self.bow_vec.transform(X_np).toarray()
        return torch.from_numpy(vectorized).float()

    @_ensure_initialized
    def train_all_v2(
        self,
        X_train_np: np.ndarray,
        Y_train: torch.Tensor,
        X_test_np: np.ndarray,
        Y_test: torch.Tensor,
    ):
        """Trains all models on the provided training data.
        :param metric_labels: a map from CVSS metric metriciations
                              to the Y_train torch.Tensor that is
                              associated with X_train.
        NOTE: this function does not work well.  Training is several orders of magnitude slower,
        and while confidence intervals are much higher, prediction results are worse.
        I suspect there is a bug in implementation.
        """
        self._validate_Y_properties(Y_train)
        X_train = self.preprocess_examples(X_train_np)
        X_test = self.preprocess_examples(X_test_np)

        if self.ipex_optimized:
            X_train = X_train.to("xpu")
            Y_train = Y_train.to("xpu")
            X_test = X_test.to("xpu")
            Y_test = Y_test.to("xpu")

        if self.cuda_enabled:
            log.debug(
                "cuda enabled; moving data + models and re-initializing optimizers"
            )
            X_train = X_train.to("cuda")
            Y_train = Y_train.to("cuda")
            X_test = X_test.to("cuda")
            Y_test = Y_test.to("cuda")
            for metric, model in self.models.items():
                model.to("cuda")
                self.optimizers[metric] = torch.optim.SGD(
                    model.parameters(), lr=self.learn_rate
                )

        start_time = time.perf_counter()
        for i, metric in enumerate(self.models.keys()):
            log.debug(f"++ training metric {i}: {metric}")
            self._train_metric_v2(X_train, Y_train[:, i], X_test, Y_test[:, i], metric)
        log.info(
            f"total training time all metrics: {time.perf_counter() - start_time:0.5}"
        )

        log.info("training complete; flushing writer")
        self.writer.flush()

    @_ensure_initialized
    def train_all(self, X_train_np: np.ndarray, Y_train: torch.Tensor):
        """Trains all models on the provided training data.
        :param metric_labels: a map from CVSS metric metriciations
                              to the Y_train torch.Tensor that is
                              associated with X_train.
        """
        self._validate_Y_properties(Y_train)
        X_train = self.preprocess_examples(X_train_np)

        # TODO: is there a better place for this?
        if self.ipex_optimized:
            X_train = X_train.to("xpu")
            Y_train = Y_train.to("xpu")

        if self.cuda_enabled:
            log.debug(
                "cuda enabled; moving data + models and re-initializing optimizers"
            )
            X_train = X_train.to("cuda")
            Y_train = Y_train.to("cuda")
            for metric, model in self.models.items():
                model.to("cuda")
                self.optimizers[metric] = torch.optim.SGD(
                    model.parameters(), lr=self.learn_rate
                )

        start_time = time.perf_counter()
        for i, metric in enumerate(self.models.keys()):
            log.debug(f"++ training metric {i}: {metric}")
            self._train_metric(X_train, Y_train[:, i], metric)
        log.info(
            f"total training time all metrics: {time.perf_counter() - start_time:0.5}"
        )

        log.info("training complete; flushing writer")
        self.writer.flush()

    @_ensure_initialized
    def predict(self, X_test_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns predictions and confidence scores
        indexed by cvss metric"""

        X = self.preprocess_examples(X_test_np)
        if self.cuda_enabled:
            log.debug("cuda enabled; moving data")
            X = X.to("cuda")

        predictions = np.zeros((X.shape[0], len(self.models)))
        confidence_scores = np.zeros((X.shape[0], len(self.models)))

        for i, (metric, model) in enumerate(self.models.items()):
            log.debug(f"predict: {metric}")
            prob = torch.nn.functional.softmax(model(X), dim=1)

            pred = torch.argmax(prob, dim=1)

            confidence = prob[range(prob.shape[0]), pred]

            predictions[:, i] = pred.cpu().numpy()
            confidence_scores[:, i] = confidence.cpu().detach().numpy()

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
        with open(os.path.join(dir_path, "bow_vec.pkl"), "wb") as f:
            pickle.dump(self.bow_vec, f)

    def load_models_full(self, models_root_dir: str):
        """Load the models into self.models and vectorizer into self.bow_vec,
        overwriting whatever was stored there."""
        for metric in CVSS_BASE_METRICS:
            self.models[metric] = torch.load(
                os.path.join(
                    self.model_save_dir, models_root_dir, f"{metric}_model.pth"
                ),
                map_location=torch.device("cuda")
                if self.cuda_enabled
                else torch.device("cpu"),
            )
        with open(
            os.path.join(self.model_save_dir, models_root_dir, "bow_vec.pkl"), "rb"
        ) as f:
            self.bow_vec = pickle.load(f)
        assert (
            self.models.keys() == CVSS_BASE_METRICS.keys()
        ), "should be no leftover metric keys"
        assert len(self.bow_vec.vocabulary_) > 0, "bow_vec should be valid"

    @_ensure_initialized
    def optimize_intel_ipex(self):
        """
        Attempt to optimize models for GPU training and inference
        using intel's ipex libraries.
        There are indeed many ways this can go wrong.

        The best documentation I can find is here:
        https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/examples.html#complete-float32-example

        Note that I do not store training data in this class,
        so it assumes that the tensors passed in have also been
        transformed via torch.tensor.to("xpu").
        TODO: Is there is a programmatic way to enforce this?
        """

        try:
            import intel_extension_for_pytorch as ipex
        except ImportError:
            log.error("could not import ibex")
            raise

        # first, optimize the criterion (this might not actually be necessary?)
        self.criterion.to("xpu")

        # then, optimize each model, passing in the optimizers
        for metric, model in self.models.items():
            log.debug(f"optimizing with ipex: {metric}")
            model.to("xpu")
            self.models[metric], _ = ipex.optimize(
                model, optimizer=self.optimizers[metric]
            )

        assert self.models.keys() == self.optimizers.keys()

        self.ipex_optimized = True
