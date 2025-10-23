# coding: utf-8

# Copyright 2025 Inria (Institut National de Recherche en Informatique
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
import warnings
from typing import (  # fmt: off
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Self,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pandas as pd
import sklearn  # type: ignore
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix  # type: ignore
from sklearn._loss.loss import (  # type: ignore
    HalfBinomialLoss,
    HalfSquaredError,
    HuberLoss,
)
from sklearn.linear_model import SGDClassifier, SGDRegressor  # type: ignore

from declearn.data_info import aggregate_data_info
from declearn.model.api import Model
from declearn.model.sklearn._np_vec import NumpyVector
from declearn.typing import Batch
from declearn.utils import DevicePolicy, register_type

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


DataArray = Union[np.ndarray, spmatrix, pd.DataFrame, pd.Series]


def select_sgd_model_dtype(
    model: Union[SGDClassifier, SGDRegressor],
    dtype: Union[str, np.dtype, Type[np.number]],
) -> str:
    """Return the current or future dtype of a scikit-learn SGD model.

    Emit a `RuntimeWarning` if the model has already been initialized
    with a dtype that does not match the input one, and/or if the
    scikit-learn version makes it impossible to enforce a non-default
    dtype.
    """
    # Identify the user-defined dtype's name.
    dtype = _get_dtype_name(dtype)
    # Warn if the dtype is already defined and does not match the argument.
    if hasattr(model, "coef_") and (model.coef_.dtype.name != dtype):
        warnings.warn(
            f"Cannot enforce dtype '{dtype}' for pre-initialized "
            f"scikit-learn SGD model (dtype '{model.coef_.dtype.name}').",
            RuntimeWarning,
            stacklevel=2,
        )
        return model.coef_.dtype.name
    # When using scikit-learn <= 1.3, warn about un-settable dtype.
    if int(sklearn.__version__.split(".", 2)[1]) < 3 and (dtype != "float64"):
        warnings.warn(
            "Using scikit-learn <1.3; hence the 'float64' dtype will"
            f"forcibly be used rather than user-input '{dtype}'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "float64"
    return dtype


def _get_dtype_name(
    dtype: Union[str, np.dtype, Type[np.number]],
) -> str:
    """Gather the name of an input numpy dtype."""
    if isinstance(dtype, np.dtype):
        return dtype.name
    if isinstance(dtype, type) and issubclass(dtype, np.number):
        return dtype.__name__
    if isinstance(dtype, str):
        return dtype
    raise TypeError("'dtype' should be a str, np.dtype or np.number type.")


@register_type(name="SklearnSGDModel", group="Model")
class SklearnSGDModel(Model):
    """Model wrapper for Scikit-Learn SGDClassifier and SGDRegressor.

    This `Model` subclass is designed to wrap a `SGDClassifier`
    or `SGDRegressor` instance (from `sklearn.linear_model`) to
    be learned federatively.

    Notes regarding device management (CPU, GPU, etc.):

    - This Model may only run on CPU, and is unaffected by device-
      management policies.
    - Calling the `update_device_policy` method has no effect, and
      raises a UserWarning if a GPU-targetting policy is passed to
      it directly.
    """

    def __init__(
        self,
        model: Union[SGDClassifier, SGDRegressor],
        dtype: Union[str, np.dtype, Type[np.number]] = "float64",
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
        dtype: str or numpy.dtype or type[np.number], default="float64"
            Data type to enforce for the model's coefficients. Input data
            will be cast to the matching dtype.
            Only used when both these conditions are met:
                - `model` is an un-initialized instance;
                  otherwise the dtype of existing coefficients is used.
                - the scikit-learn version is >= 1.3;
                  otherwise the dtype is forced to float64
        """
        if not isinstance(model, (SGDClassifier, SGDRegressor)):
            raise TypeError(
                "'model' should be a scikit-learn SGDClassifier"
                " or SGDRegressor instance."
            )
        model = model.set_params(
            eta0=1.0,
            learning_rate="constant",
            warm_start=False,
            average=False,
        )
        super().__init__(model)
        self._dtype = select_sgd_model_dtype(model, dtype)
        self._predict = (
            self._model.decision_function
            if isinstance(model, SGDClassifier)
            else self._model.predict
        )
        self._loss_fn: Optional[
            Callable[[np.ndarray, np.ndarray], np.ndarray]
        ] = None

    @property
    def device_policy(
        self,
    ) -> DevicePolicy:
        return DevicePolicy(gpu=False, idx=None)

    @property
    def required_data_info(
        self,
    ) -> Set[str]:
        if hasattr(self._model, "coef_"):
            return set()
        if isinstance(self._model, SGDRegressor):
            return {"features_shape"}
        return {"features_shape", "classes"}

    def initialize(
        self,
        data_info: Dict[str, Any],
    ) -> None:
        # Skip for pre-initialized models.
        if hasattr(self._model, "coef_"):
            return
        # Check that required fields are available and of valid type.
        data_info = aggregate_data_info([data_info], self.required_data_info)
        if not (
            len(data_info["features_shape"]) == 1
            and isinstance(data_info["features_shape"][0], int)
        ):
            raise ValueError(
                "SklearnSGDModel requires fixed-size 1-d input features."
            )
        feat = data_info["features_shape"][0]
        # SGDClassifier case.
        if isinstance(self._model, SGDClassifier):
            self._model.classes_ = np.array(list(data_info["classes"]))
            n_classes = len(self._model.classes_)
            dim = n_classes if (n_classes > 2) else 1
            self._model.coef_ = np.zeros((dim, feat), dtype=self._dtype)
            self._model.intercept_ = np.zeros((dim,), dtype=self._dtype)
        # SGDRegressor case.
        else:
            self._model.coef_ = np.zeros((feat,), dtype=self._dtype)
            self._model.intercept_ = np.zeros((1,), dtype=self._dtype)

    # pylint: disable=too-many-positional-arguments
    @classmethod
    def from_parameters(  # noqa: PLR0913
        cls,
        kind: Literal["classifier", "regressor"],
        loss: Optional[LossesLiteral] = None,
        penalty: Literal["none", "l1", "l2", "elasticnet"] = "l2",
        alpha: float = 1e-4,
        l1_ratio: float = 0.15,
        epsilon: float = 0.1,
        fit_intercept: bool = True,
        n_jobs: Optional[int] = None,
        dtype: Union[str, np.dtype, Type[np.number]] = "float64",
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
        penalty: {"none", "l1", "l2", "elasticnet"}, default="l2"
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
        dtype: str or numpy.dtype or type[np.number], default="float64"
            Data type to enforce for the model's coefficients. Input data
            will be cast to the matching dtype.
            Only used when both these conditions are met:
                - `model` is an un-initialized instance;
                  otherwise the dtype of existing coefficients is used.
                - the scikit-learn version is >= 1.3;
                  otherwise the dtype is forced to float64

        Notes
        -----
        - Save for `kind`, all parameters are strictly equivalent to those
          of `sklearn.linear_modelSGDClassifier` and `SGDRegressor`. Refer
          to the latter' documentation for additional details.
        - Note that unexposed parameters from those classes are simply not
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
        # Invalid input case.
        else:  # pragma: no cover
            raise ValueError(
                "Invalid value for SklearnSGDModel 'kind': must be one of "
                f"{'classifier', 'regressor'}, received '{kind}'."
            )
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
        return cls(model, dtype)

    def get_config(
        self,
    ) -> Dict[str, Any]:
        is_clf = isinstance(self._model, SGDClassifier)
        data_info: Optional[Dict[str, Any]] = None
        if hasattr(self._model, "coef_"):
            data_info = {
                "features_shape": (self._model.coef_.shape[-1],),
                "classes": self._model.classes_.tolist() if is_clf else None,
            }
            dtype = self._model.coef_.dtype.name
        else:
            dtype = self._dtype
        return {
            "kind": "classifier" if is_clf else "regressor",
            "params": self._model.get_params(),
            "data_info": data_info,
            "dtype": dtype,
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
            skmod = SGDClassifier(**config["params"])
        else:
            skmod = SGDRegressor(**config["params"])
        model = cls(skmod, dtype=config.get("dtype", "float64"))
        if config.get("data_info"):
            model.initialize(config["data_info"])
        return model

    def get_weights(
        self,
        trainable: bool = False,
    ) -> NumpyVector:
        weights = {
            "intercept": self._model.intercept_.copy(),
            "coef": self._model.coef_.copy(),
        }
        return NumpyVector(weights)

    def set_weights(
        self,
        weights: NumpyVector,
        trainable: bool = False,
    ) -> None:
        if not isinstance(weights, NumpyVector):
            raise TypeError("SklearnSGDModel requires NumpyVector weights.")
        for key in ("coef", "intercept"):
            if key not in weights.coefs:
                raise KeyError(
                    f"Missing required '{key}' in the received vector."
                )
        self._model.coef_ = weights.coefs["coef"].astype(self._dtype)
        self._model.intercept_ = weights.coefs["intercept"].astype(self._dtype)

    def compute_batch_gradients(
        self,
        batch: Batch,
        max_norm: Optional[float] = None,
    ) -> NumpyVector:
        # Unpack, validate and repack input data.
        x_data, y_data, s_wght = self._unpack_batch(batch)
        # Iteratively compute sample-wise gradients.
        grad = [
            self._compute_sample_gradient(x, y)
            for x, y in zip(x_data, y_data, strict=False)  # type: ignore
        ]
        # Optionally clip sample-wise gradients based on their L2 norm.
        if max_norm:
            for vec in grad:
                for key, arr in vec.coefs.items():
                    norm = np.sqrt(np.sum(np.square(arr)))
                    # update arr (without mutating loop variables)
                    vec.coefs[key] *= min(max_norm / norm, 1)
        # Optionally re-weight gradients based on sample weights.
        if s_wght is not None:
            grad = [g * w for g, w in zip(grad, s_wght, strict=False)]  # type: ignore
        # Compute and record the loss value on the entire batch.
        loss = self.loss_function(
            y_data,  # type: ignore
            self._predict(x_data),
        )
        self._loss_history.append(float(loss.mean()))
        # Batch-average the gradients and return them.
        return sum(grad) / len(grad)  # type: ignore

    def _unpack_batch(
        self,
        batch: Batch,
    ) -> Tuple[DataArray, DataArray, Optional[DataArray]]:
        """Verify and unpack an input batch into (x, y, [w]).

        Note: this method does not verify arrays' dimensionality or
        shape coherence; the wrapped sklearn objects already do so.
        """
        x_data, y_data, s_wght = batch
        if (
            (y_data is None)
            or isinstance(y_data, list)
            or isinstance(x_data, list)
        ):
            raise TypeError(
                "'SklearnSGDModel' requires (array, array, [array|None]) "
                "data batches."
            )
        x_data = self._validate_and_cast_array(x_data)
        y_data = self._validate_and_cast_array(y_data)
        if s_wght is not None:
            s_wght = self._validate_and_cast_array(s_wght)
        return x_data, y_data, s_wght

    def _validate_and_cast_array(
        self,
        array: ArrayLike,
    ) -> DataArray:
        """Type-check and type-cast an input data array."""
        if not isinstance(array, typing.get_args(DataArray)):
            raise TypeError(
                f"Invalid data type for 'SklearnSGDModel': '{type(array)}'."
            )
        return array.astype(self._dtype)  # type: ignore

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
        return w_srt - w_end

    def apply_updates(
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
        """Return a function to compute point-wise loss for a given batch.

        Warning : this method use sklearn SGDRegressor / SGDClassifier internal
        mechanisms (and not public API) to instantiate losses (i.e.
        using Cython loss classes in the process). Those mechanisms have
        already changed in the past, breaking some stuff, so it can happen
        again in the future
        """
        # Losses in the following conditions need to be retrieved explicitly
        # (cannot use a "py_loss" of the loss Cython class)
        if self._model.loss == "squared_error":

            def loss_1d(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
                return HalfSquaredError().loss(
                    y_true=y_true, raw_prediction=y_pred
                )

        elif self._model.loss == "huber":

            def loss_1d(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
                return HuberLoss(delta=self._model.epsilon).loss(
                    y_true=y_true, raw_prediction=y_pred
                )

        elif self._model.loss == "log_loss":

            def loss_1d(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
                return HalfBinomialLoss().loss(
                    y_true=y_true, raw_prediction=y_pred
                )

        # Other losses can be retrieved via "loss_functions" dict and
        # should implement a "py_loss" attribute
        else:
            # fmt: off
            # Instantiate a loss function from the wrapped model's specs.
            loss_cls, *args = self._model.loss_functions[self._model.loss]
            if self._model.loss in (
                "huber", "epsilon_insensitive", "squared_epsilon_insensitive"
            ):
                args = (self._model.epsilon,)

            loss_obj = loss_cls(*args)
            # Check that loss class has attribute "py_loss"
            if not hasattr(loss_obj, "py_loss"):
                raise NotImplementedError(
                    f"Loss {self._model.loss} not supported : "
                    "corresponding py_loss function does not exist"
                )
            loss_smp = loss_obj.py_loss
            # Wrap it to support batched inputs.
            def loss_1d(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
                return np.array(
                    [loss_smp(*smp) for smp in zip(y_pred, y_true, strict=False)]
                )

        # For multiclass classifiers, further wrap to support 2d predictions.
        if len(getattr(self._model, "classes_", [])) > 2:

            def to_float_contig(arr: np.ndarray) -> np.ndarray:
                """Convert array to float64 and C-contiguous layout"""
                return np.ascontiguousarray(arr, dtype=np.float64)

            def loss_fn(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
                return np.sum(
                    [
                        loss_1d(
                            to_float_contig(y_true == class_val),
                            to_float_contig(y_pred[:, i]),
                        )
                        for i, class_val in enumerate(self._model.classes_)
                    ],
                    axis=0,
                )

        else:
            loss_fn = loss_1d
        return loss_fn
        # fmt: on

    def update_device_policy(
        self,
        policy: Optional[DevicePolicy] = None,
    ) -> None:
        if policy is not None and policy.gpu:
            warnings.warn(
                "'SklearnSGDModel' only runs on a CPU backend.",
                stacklevel=2,
            )
