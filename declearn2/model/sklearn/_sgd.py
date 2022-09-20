# coding: utf-8

"""Model subclass to wrap scikit-learn SGD classifier and regressor models."""

import typing
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import SGDClassifier, SGDRegressor  # type: ignore
from typing_extensions import Literal  # future: import from typing (Py>=3.8)

from declearn2.model.api import Model, NumpyVector
from declearn2.typing import Batch


LossesLiteral = Literal[
    'hinge', 'log_loss', 'modified_huber', 'squared_hinge',
    'perceptron', 'squared_error', 'huber', 'epsilon_insensitive',
    'squared_epsilon_insensitive'
]
REG_LOSSES = (
    'squared_error', 'huber', 'epsilon_insensitive',
    'squared_epsilon_insensitive'
)


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

        Note: See `SklearnSGDModel.from_parameters` for an alternative
              constructor that does not require a manual instantiation
              of the wrapped scikit-learn model.

        Arguments:
        ---------
        model: SGDClassifier or SGDRegressor
            Scikit-learn model that needs wrapping for federated training.
            Note that some hyperparameters will be overridden, as will the
            model's existing weights.
        n_features: int
            Number of input features in the learning task.
        n_classes: int or None, default=None
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
        """Initialize a sklearn SGD model's weights.

        Create (or overwrite) `coef_` and `intercept_` attributes
        and set them to zero-valued arrays (as is done by sklearn
        the first time (partial_)fit is called).
        If the model is a classifier, also set `classes_`.
        """
        if isinstance(model, SGDClassifier):
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
            # Assign attributes.
            model.classes_ = np.array(classes)
            model.coef_ = np.zeros(
                (n_classes if (n_classes > 2) else 1, n_features)
            )
            model.intercept_ = np.zeros((n_classes,))
        else:
            # Assign attributes in the SGDRegressor case.
            model.coef_ = np.zeros((n_features,))
            model.intercept_ = np.zeros((1,))

    @classmethod
    def from_parameters(
            cls,
            n_features: int,
            n_classes: Optional[int] = None,
            classes: Optional[Union[np.ndarray, List[int]]] = None,
            loss: Optional[LossesLiteral] = None,
            penalty: Literal['l1', 'l2', 'elasticnet'] = 'l2',
            alpha: float = 1e-4,
            l1_ratio: float = 0.15,
            epsilon: float = 0.1,
            fit_intercept: bool = True,
            n_jobs: Optional[int] = None,
        ) -> 'SklearnSGDModel':
        """Instantiate a SklearnSGDModel from model parameters.

        This classmethod is an alternative constructor to instantiate
        a SklearnSGDModel without first instantiating a scikit-learn
        SGDRegressor or SGDClassifier.

        Arguments:
        ---------
        * The first three arguments are from `SklearnSGDModel.__init__`
          and specify the task at hand (regression or classification)
          as well as the model's dimensionality.
        * All other arguments come from the scikit-learn `SGDRegressor`
          and `SGDClassifier` classes (from `sklearn.linear_model`),
          and specify the loss function (hence type of model) as well
          as some of its parameters.
        * Note that unexposed parameters from the scikit-learn classes
          are simply of no use when wrapped by `SklearnSGDModel`.
        """
        # partially-inherited signature; pylint: disable=too-many-arguments
        # SGDClassifier case.
        if n_classes or (classes is not None):
            loss = loss or 'hinge'
            if loss not in typing.get_args(LossesLiteral):
                raise ValueError(f"Invalid loss '{loss}' for SGDClassifier.")
            sk_cls = SGDClassifier
        # SGDRegressor case.
        else:
            loss = loss or 'squared_error'
            if loss not in REG_LOSSES:
                raise ValueError(f"Invalid loss '{loss}' for SGDRegressor.")
            sk_cls = SGDRegressor
        # Instantiate the sklearn model, wrap it up and return.
        model = sk_cls(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            epsilon=epsilon, fit_intercept=fit_intercept, n_jobs=n_jobs,
        )
        return cls(model, n_features, n_classes, classes)

    def get_config(
            self,
        ) -> Dict[str, Any]:
        """Return the model's parameters as a JSON-serializable dict."""
        is_clf = isinstance(self._model, SGDClassifier)
        data_specs = {
            "n_features": self._model.coef_.shape[-1],
            "n_classes": len(self._model.classes_) if is_clf else None,
            "classes": self._model.classes_.tolist() if is_clf else None,
        }
        return {
            "is_clf": is_clf,
            "params": self._model.get_params(),
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
        m_cls = SGDClassifier if config["is_clf"] else SGDRegressor
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
            batch: Batch,
        ) -> NumpyVector:
        """Compute and return the model's gradients over a data batch."""
        # Unpack, validate and repack input data.
        x_data, y_data, s_wght = self._verify_batch(batch)
        data = (x_data, y_data) if s_wght is None else (x_data, y_data, s_wght)
        # Iteratively compute sample-wise gradients. Average them and return.
        grad = [
            self._compute_sample_gradient(*smp)
            for smp in zip(*data)  # type: ignore
        ]
        return sum(grad) / len(grad)  # type: ignore

    def _verify_batch(
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

    def compute_loss(
            self,
            dataset: Iterable[Batch],
        ) -> float:
        """Compute the average loss of the model on a given dataset.

        dataset: iterable of batches
            Iterable yielding batch structures that are to be unpacked
            into (input_features, target_labels, [sample_weights]).
            If set, sample weights will affect the loss averaging.

        Return the average value of the model's loss over samples.
        """
        # TODO: implement SklearnMetric objects and abstract this code
        # Instantiate a loss function from the wrapped model's specs.
        if hasattr(self._model, 'loss_function_'):
            loss_fn = self._model.loss_function_.py_loss
        else:
            loss_cls, *args = self._model.loss_functions[self._model.loss]
            loss_fn = loss_cls(*args).py_loss
        # Initialize loss numerator and denominator.
        loss = 0
        nsmp = 0
        # Iteratively compute and accumulate batch- and sample-wise loss.
        for batch in dataset:
            inputs, y_true, s_wght = self._verify_batch(batch)
            y_pred = self._model.predict(inputs)
            if s_wght is None:
                loss += sum(
                    loss_fn(*smp)
                    for smp in zip(y_pred, y_true)  # type: ignore
                )
                nsmp += len(y_pred)
            else:
                loss += sum(
                    smp[2] * loss_fn(smp[0], smp[1])
                    for smp in zip(y_pred, y_true, s_wght)  # type: ignore
                )
                nsmp += np.sum(s_wght)
        # Reduce the results and return them.
        return loss / nsmp
