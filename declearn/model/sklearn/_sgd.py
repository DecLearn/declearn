# coding: utf-8

# Copyright 2023 Inria (Institut National de Recherche en Informatique
# et Automatique)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model subclass to wrap scikit-learn SGD classifier and regressor models."""

import typing
from typing import Any, Callable, Dict, Literal, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import SGDClassifier, SGDRegressor  # type: ignore
from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.data_info import aggregate_data_info
from declearn.model.api import Model
from declearn.model.sklearn._np_vec import NumpyVector
from declearn.typing import Batch
from declearn.utils import register_type


__all__ = [
    "SklearnSGDModel",
]


LossesLiteral = Literal[
    "hinge",
    "log_loss",
    "modified_huber",
    "squared_hinge",
    "perceptron",
    "squared_error",
    "huber",
    "epsilon_insensitive",
    "squared_epsilon_insensitive",
]
REG_LOSSES = (
    "squared_error",
    "huber",
    "epsilon_insensitive",
    "squared_epsilon_insensitive",
)


@register_type(name="SklearnSGDModel", group="Model")
class SklearnSGDModel(Model):
    """Model wrapper for Scikit-Learn SGDClassifier and SGDRegressor.

    This `Model` subclass is designed to wrap a `SGDClassifier`
    or `SGDRegressor` instance (from `sklearn.linear_model`) to
    be learned federatively.
    """

    def __init__(
        self,
        model: Union[SGDClassifier, SGDRegressor],
    ) -> None:
        """Instantiate a Model interfacing a sklearn SGD-based model.

        Note: See `SklearnSGDModel.from_parameters` for an alternative
              constructor that does not require a manual instantiation
              of the wrapped scikit-learn model.

        Parameters
        ----------
        model: SGDClassifier or SGDRegressor
            Scikit-learn model that needs wrapping for federated training.
            Note that some hyperparameters will be overridden, as will the
            model's existing weights (if any).
        """
        if not isinstance(model, (SGDClassifier, SGDRegressor)):
            raise TypeError(
                "'model' should be a scikit-learn SGDClassifier"
                " or SGDRegressor instance."
            )
        model = model.set_params(
            eta0=0.1,
            learning_rate="constant",
            warm_start=False,
            average=False,
        )
        super().__init__(model)
        self._initialized = False
        self._predict = (
            self._model.decision_function
            if isinstance(model, SGDClassifier)
            else self._model.predict
        )
        self._loss_fn = (
            None
        )  # type: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]

    @property
    def required_data_info(
        self,
    ) -> Set[str]:
        if isinstance(self._model, SGDRegressor):
            return {"n_features"}
        return {"n_features", "classes"}

    def initialize(
        self,
        data_info: Dict[str, Any],
    ) -> None:
        # Check that required fields are available and of valid type.
        data_info = aggregate_data_info([data_info], self.required_data_info)
        # SGDClassifier case.
        if isinstance(self._model, SGDClassifier):
            self._model.classes_ = np.array(list(data_info["classes"]))
            n_classes = len(self._model.classes_)
            dim = n_classes if (n_classes > 2) else 1
            self._model.coef_ = np.zeros((dim, data_info["n_features"]))
            self._model.intercept_ = np.zeros((dim,))
        # SGDRegressor case.
        else:
            self._model.coef_ = np.zeros((data_info["n_features"],))
            self._model.intercept_ = np.zeros((1,))
        # Mark the SklearnSGDModel as initialized.
        self._initialized = True

    @classmethod
    def from_parameters(
        cls,
        kind: Literal["classifier", "regressor"],
        loss: Optional[LossesLiteral] = None,
        penalty: Literal["l1", "l2", "elasticnet"] = "l2",
        alpha: float = 1e-4,
        l1_ratio: float = 0.15,
        epsilon: float = 0.1,
        fit_intercept: bool = True,
        n_jobs: Optional[int] = None,
    ) -> Self:
        """Instantiate a SklearnSGDModel from model parameters.

        This classmethod is an alternative constructor to instantiate
        a SklearnSGDModel without first instantiating a scikit-learn
        SGDRegressor or SGDClassifier.

        Parameters
        ----------
        kind: "classifier" or "regressor"
            Literal string specifying the task-based kind of model to use.
        loss: str or None, default=None
            The loss function to be used.
            See `sklearn.linear_model.SGDRegressor` and `SGDClassifier`
            documentation for details on possible values. If None, set
            to "hinge" for classifier or "squared_error" for regressor.
        penalty: {"l1", "l2", "elasticnet"}, default="l2"
            The penalty (i.e. regularization term) to be used.
        alpha: float, default=0.0001
            Regularization constant (the higher the stronger).
            Alpha must be in [0, inf[ and is constant through training.
        l1_ratio: float, default=0.15
            Mixing parameter for elasticnet regularization.
            Only used if `penalty="elasticnet"`. Must be in [0, 1].
        epsilon: float, default=0.1
            Epsilon in the epsilon-insensitive loss functions. For these,
            defines an un-penalized margin of error. Must be in [0, inf[.
        fit_intercept: bool, default=True
            Whether an intercept should be estimated or not.
        n_jobs: int or None, default=None
            Number of CPUs to use when to compute one-versus-all.
            Only used for multi-class classifiers.
            `None` means 1, while -1 means all available CPUs.

        Notes
        -----
        Save for `kind`, all parameters are strictly equivalent to those
        of `sklearn.linear_modelSGDClassifier` and `SGDRegressor`. Refer
        to the latter' documentation for additional details.
        Note that unexposed parameters from those classes are simply not
        used and/or overwritten when wrapped by `SklearnSGDModel`.

        Returns
        -------
        model: SklearnSGDModel
            A declearn Model wrapping an instantiated scikit-learn one.
        """
        # partially-inherited signature; pylint: disable=too-many-arguments
        kwargs = {}
        # SGDClassifier case.
        if kind == "classifier":
            loss = loss or "hinge"
            if loss not in typing.get_args(LossesLiteral):
                raise ValueError(f"Invalid loss '{loss}' for SGDClassifier.")
            sk_cls = SGDClassifier
            kwargs["n_jobs"] = n_jobs
        # SGDRegressor case.
        elif kind == "regressor":
            loss = loss or "squared_error"
            if loss not in REG_LOSSES:
                raise ValueError(f"Invalid loss '{loss}' for SGDRegressor.")
            sk_cls = SGDRegressor
        # Instantiate the sklearn model, wrap it up and return.
        model = sk_cls(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            epsilon=epsilon,
            fit_intercept=fit_intercept,
            **kwargs,
        )
        return cls(model)

    def get_config(
        self,
    ) -> Dict[str, Any]:
        is_clf = isinstance(self._model, SGDClassifier)
        data_info = None  # type: Optional[Dict[str, Any]]
        if self._initialized:
            data_info = {
                "n_features": self._model.coef_.shape[-1],
                "classes": self._model.classes_.tolist() if is_clf else None,
            }
        return {
            "kind": "classifier" if is_clf else "regressor",
            "params": self._model.get_params(),
            "data_info": data_info,
        }

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        """Instantiate a SklearnSGDModel from a configuration dict."""
        for key in ("kind", "params"):
            if key not in config:
                raise KeyError(f"Missing key '{key}' in the config dict.")
        if config["kind"] == "classifier":
            model = cls(SGDClassifier(**config["params"]))
        else:
            model = cls(SGDRegressor(**config["params"]))
        if config.get("data_info"):
            model.initialize(config["data_info"])
        return model

    def get_weights(
        self,
    ) -> NumpyVector:
        weights = {
            "intercept": self._model.intercept_.copy(),
            "coef": self._model.coef_.copy(),
        }
        return NumpyVector(weights)

    def set_weights(  # type: ignore  # Vector subtype specification
        self,
        weights: NumpyVector,
    ) -> None:
        if not isinstance(weights, NumpyVector):
            raise TypeError("SklearnSGDModel requires NumpyVector weights.")
        for key in ("coef", "intercept"):
            if key not in weights.coefs:
                raise TypeError(
                    f"Missing required '{key}' in the received vector."
                )
        self._model.coef_ = weights.coefs["coef"].copy()
        self._model.intercept_ = weights.coefs["intercept"].copy()

    def compute_batch_gradients(
        self,
        batch: Batch,
        max_norm: Optional[float] = None,
    ) -> NumpyVector:
        # Unpack, validate and repack input data.
        x_data, y_data, s_wght = self._unpack_batch(batch)
        # Iteratively compute sample-wise gradients.
        grad = [
            self._compute_sample_gradient(x, y)  # type: ignore
            for x, y in zip(x_data, y_data)  # type: ignore
        ]
        # Optionally clip sample-wise gradients based on their L2 norm.
        if max_norm:
            for vec in grad:
                for arr in vec.coefs.values():
                    norm = np.sqrt(np.sum(np.square(arr)))
                    arr *= min(max_norm / norm, 1)
        # Optionally re-weight gradients based on sample weights.
        if s_wght is not None:
            grad = [g * w for g, w in zip(grad, s_wght)]  # type: ignore
        # Batch-average the gradients and return them.
        return sum(grad) / len(grad)  # type: ignore

    def _unpack_batch(
        self,
        batch: Batch,
    ) -> Tuple[ArrayLike, ArrayLike, Optional[ArrayLike]]:
        """Verify and unpack an input batch into (x, y, [w]).

        Note: this method does not verify arrays' dimensionality or
        shape coherence; the wrapped sklearn objects already do so.
        """
        x_data, y_data, s_wght = batch
        invalid = (y_data is None) or isinstance(y_data, list)
        if invalid or isinstance(x_data, list):
            raise TypeError(
                "'SklearnSGDModel' requires (array, array, [array|None]) "
                "data batches."
            )
        return x_data, y_data, s_wght  # type: ignore

    def _compute_sample_gradient(
        self,
        x_smp: ArrayLike,
        y_smp: float,
    ) -> NumpyVector:
        """Compute and return the model's gradients over a single sample."""
        # Gather current weights.
        w_srt = self.get_weights()
        # Perform SGD step and gather weights.
        x_smp = x_smp.reshape((1, -1))  # type: ignore
        self._model.partial_fit(x_smp, [y_smp])
        w_end = self.get_weights()
        # Restore the model's weights.
        self.set_weights(w_srt)
        # Compute gradients based on weights' update.
        return (w_srt - w_end) / self._model.eta0

    def apply_updates(  # type: ignore  # Vector subtype specification
        self,
        updates: NumpyVector,
    ) -> None:
        if not isinstance(updates, NumpyVector):
            raise TypeError("SklearnSGDModel requires NumpyVector updates.")
        self._model.coef_ += updates.coefs["coef"]
        self._model.intercept_ += updates.coefs["intercept"]

    def compute_batch_predictions(
        self,
        batch: Batch,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        inputs, y_true, s_wght = self._unpack_batch(batch)
        y_pred = self._predict(inputs)
        return y_true, y_pred, s_wght  # type: ignore

    def loss_function(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        if self._loss_fn is None:
            self._loss_fn = self._setup_loss_fn()
        return self._loss_fn(y_true, y_pred)

    def _setup_loss_fn(
        self,
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Return a function to compute point-wise loss for a given batch."""
        # fmt: off
        # Gather or instantiate a loss function from the wrapped model's specs.
        if hasattr(self._model, "loss_function_"):
            loss_smp = self._model.loss_function_.py_loss
        else:
            loss_cls, *args = self._model.loss_functions[self._model.loss]
            loss_smp = loss_cls(*args).py_loss
        # Wrap it to support batched inputs.
        def loss_1d(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
            return np.array([loss_smp(*smp) for smp in zip(y_pred, y_true)])
        # For multiclass classifiers, further wrap to support 2d predictions.
        if len(getattr(self._model, "classes_", [])) > 2:
            def loss_fn(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
                return np.sum([
                    loss_1d(y_true == val, y_pred[:, i])
                    for i, val in enumerate(self._model.classes_)
                ], axis=0)
        else:
            loss_fn = loss_1d
        return loss_fn
