# coding: utf-8

"""Model subclass to wrap scikit-learn SGD classifier and regressor models."""

import typing
from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import SGDClassifier, SGDRegressor  # type: ignore
from typing_extensions import Literal  # future: import from typing (Py>=3.8)

from declearn2.data_info import aggregate_data_info
from declearn2.model.api import Model, NumpyVector
from declearn2.typing import Batch
from declearn2.utils import register_type


LossesLiteral = Literal[
    'hinge', 'log_loss', 'modified_huber', 'squared_hinge',
    'perceptron', 'squared_error', 'huber', 'epsilon_insensitive',
    'squared_epsilon_insensitive'
]
REG_LOSSES = (
    'squared_error', 'huber', 'epsilon_insensitive',
    'squared_epsilon_insensitive'
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
            learning_rate='constant',
            warm_start=False,
            average=False,
        )
        super().__init__(model)
        self._initialized = False

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
            kind: Literal['classifier', 'regressor'],
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
            loss = loss or 'hinge'
            if loss not in typing.get_args(LossesLiteral):
                raise ValueError(f"Invalid loss '{loss}' for SGDClassifier.")
            sk_cls = SGDClassifier
            kwargs["n_jobs"] = n_jobs
        # SGDRegressor case.
        elif kind == "regressor":
            loss = loss or 'squared_error'
            if loss not in REG_LOSSES:
                raise ValueError(f"Invalid loss '{loss}' for SGDRegressor.")
            sk_cls = SGDRegressor
        # Instantiate the sklearn model, wrap it up and return.
        model = sk_cls(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            epsilon=epsilon, fit_intercept=fit_intercept, **kwargs
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
        ) -> 'SklearnSGDModel':
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
        return NumpyVector({
            "intercept": self._model.intercept_.copy(),
            "coef": self._model.coef_.copy(),
        })

    def set_weights(
            self,
            weights: NumpyVector,
        ) -> None:
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
        self._model.coef_ += updates.coefs["coef"]
        self._model.intercept_ += updates.coefs["intercept"]

    def compute_loss(
            self,
            dataset: Iterable[Batch],
        ) -> float:
        """Compute the average loss of the model on a given dataset.

        Parameters
        ----------
        dataset: iterable of batches
            Iterable yielding batch structures that are to be unpacked
            into (input_features, target_labels, [sample_weights]).
            If set, sample weights will affect the loss averaging.

        Returns
        -------
        loss: float
            Average value of the model's loss over samples.
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
