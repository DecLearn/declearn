# coding: utf-8

"""Model subclass to wrap scikit-learn SGD classifier and regressor models."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import SGDClassifier, SGDRegressor  # type: ignore

from declearn2.model.api import Model, NumpyVector
from declearn2.utils import unpack_batch


class SklearnSGDModel(Model):
    """Model wrapper for Scikit-Learn SGDClassifier and SGDRegressor.

    This `Model` subclass is designed to wrap a `SGDClassifier`
    or `SGDRegressor` instance (from `sklearn.linear_model`) to
    be learned federatively.
    """

    def __init__(
            self,
            model: Union[SGDClassifier, SGDRegressor],
            n_features: int,
            n_classes: Optional[int] = None,
        ) -> None:
        """Instantiate a Model interface wrapping a 'model' object."""
        if not isinstance(model, (SGDClassifier, SGDRegressor)):
            raise TypeError(
                "'model' should be a scikit-learn SGDClassifier"
                " or SGDRegressor instance."
            )
        model = model.set_params(
            eta0=0.1,
            learning_rate='constant',
            warm_start=False,
            average=False,
        )
        self._initialize_coefs(model, n_features, n_classes)
        super().__init__(model)

    @staticmethod
    def _initialize_coefs(
            model: Union[SGDClassifier, SGDRegressor],
            n_features: int,
            n_classes: Optional[int] = None,
        ) -> None:
        """Initialize a sklearn SGD model's weights."""
        if isinstance(model, SGDRegressor):
            coef = np.zeros((n_features,))
            intc = np.zeros((1,))
        elif n_classes:
            coef = np.zeros((n_features, n_classes))
            intc = np.zeros((n_classes,))
        else:
            raise ValueError(
                "'n_classes' is required to initialize a SGDClassifier"
            )
        model.coef_ = coef
        model.intercept_ = intc

    def get_config(
            self,
        ) -> Dict[str, Any]:
        """Return the model's parameters as a JSON-serializable dict."""
        is_clf = isinstance(self._model, SGDClassifier)
        data_specs = {
            "n_features": self._model.coef_.shape[0],
            "n_classes": self._model.coef_.shape[1] if is_clf else None,
        }
        return {
            "is_clf": is_clf,
            "params": self._model.get_config(),
            "kwargs": data_specs,
        }

    @classmethod
    def from_config(
            cls,
            config: Dict[str, Any],
        ) -> 'Model':
        """Instantiate a model from a configuration dict."""
        for key in ("is_clf", "params", "kwargs"):
            if key not in config:
                raise KeyError(f"Missing key '{key}' in the config dict.")
        m_cls = SGDClassifier if config["classif"] else SGDRegressor
        model = m_cls(**config["params"])
        return cls(model, **config["kwargs"])

    def get_weights(
            self,
        ) -> NumpyVector:
        """Return the model's trainable weights."""
        return NumpyVector({
            "intercept": self._model.intercept_.copy(),
            "coef": self._model.coef_.copy(),
        })

    def set_weights(
            self,
            weights: NumpyVector,
        ) -> None:
        """Assign values to the model's trainable weights."""
        for key in ("coef", "intercept"):
            if key not in weights.coefs:
                raise TypeError(
                    f"Missing required '{key}' in the received vector."
                )
        self._model.coef_ = weights.coefs["coef"]
        self._model.intercept_ = weights.coefs["intercept"]

    def compute_batch_gradients(
            self,
            batch: Union[ArrayLike, List[Optional[ArrayLike]]],
        ) -> NumpyVector:
        """Compute and return the model's gradients over a data batch."""
        # Unpack the batch data, checking it has proper specifications.
        x_data, y_data, s_wght = unpack_batch(batch)
        if y_data is None:
            raise TypeError(
                "'SklearnSGDModel' requires batches to contain target data."
            )
        # Iteratively compute sample-wise gradients. Average and return.
        data = [arr for arr in (x_data, y_data, s_wght) if arr is not None]
        grad = [
            self._compute_sample_gradient(*smp)
            for smp in zip(*data)  # type: ignore
        ]
        return sum(grad) / len(grad)  # type: ignore

    def _compute_sample_gradient(
            self,
            x_smp: ArrayLike,
            y_smp: float,
            s_wgt: Optional[float] = None,
        ) -> NumpyVector:
        """Compute and return the model's gradients over a single sample."""
        # Gather current weights.
        w_srt = self.get_weights()
        # Perform SGD step and gather weights.
        x_smp = x_smp.reshape((1, -1))  # type: ignore
        s_wgt = None if s_wgt is None else [s_wgt]  # type: ignore
        self._model.partial_fit(x_smp, [y_smp], sample_weight=s_wgt)
        w_end = self.get_weights()
        # Restore the model's weights.
        self.set_weights(w_srt)
        # Compute gradients based on weights' update.
        return (w_srt - w_end) / self._model.eta0  # type: ignore

    def apply_updates(  # type: ignore  # future: revise
            self,
            updates: NumpyVector,
        ) -> None:
        """Apply updates to the model's weights."""
        self._model.coef_ += updates.coefs["coef"]
        self._model.intercept_ += updates.coefs["intercept"]
