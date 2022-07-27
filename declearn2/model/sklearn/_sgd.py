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
            classes: Optional[Union[np.ndarray, List[int]]] = None,
        ) -> None:
        """Instantiate a Model interfacing a sklearn SGD-based model.

        model: SGDClassifier or SGDRegressor
            Scikit-learn model that needs wrapping for federated training.
            Note that some hyperparameters will be overridden, as will the
            model's existing weights.
        n_features: int
            Number of input features in the learning task.
        n_classes: int ot None, default=None
            If `model` is a `SGDClassifier`, number of target classes.
            May be inferred from `classes` if provided.
        classes: np.ndarray or list[int] or None, default=None
            If `model` is a `SGDClassifier`, values of target classes.
            If None, will be set to `np.arange(n_classes)`.
        """
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
        self._initialize_coefs(model, n_features, n_classes, classes)
        super().__init__(model)

    @staticmethod
    def _initialize_coefs(
            model: Union[SGDClassifier, SGDRegressor],
            n_features: int,
            n_classes: Optional[int] = None,
            classes: Optional[Union[np.ndarray, List[int]]] = None,
        ) -> None:
        """Initialize a sklearn SGD model's weights."""
        if isinstance(model, SGDClassifier):
            # Cross-check or harmonize classes and n_classes.
            if n_classes is None:
                if classes is None:
                    raise ValueError(
                        "At least one of 'n_classes' or 'classes' must be "
                        "specified to initialize a SGDClassifier model."
                    )
                n_classes = len(classes)
            elif classes is None:
                classes = np.arange(n_classes)
            elif len(classes) != n_classes:
                raise ValueError(
                    f"'n_classes' is {n_classes} but "
                    f"'classes' has length {len(classes)}."
                )
            # Assign 'classes_' attribute to avoid partial_fit failure.
            model.classes_ = np.array(classes)
        else:
            n_classes = 1  # single-target regression
        # Assign zero-valued coefficients. Note: sklearn backend does the same.
        model.coef_ = np.zeros((n_features,))
        model.intercept_ = np.zeros((n_classes,))

    def get_config(
            self,
        ) -> Dict[str, Any]:
        """Return the model's parameters as a JSON-serializable dict."""
        is_clf = isinstance(self._model, SGDClassifier)
        data_specs = {
            "n_features": self._model.coef_.shape[0],
            "n_classes": self._model.coef_.shape[1] if is_clf else None,
            "classes": self._model.classes_.tolist() if is_clf else None,
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
