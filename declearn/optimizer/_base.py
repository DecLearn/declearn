# coding: utf-8

"""Base class to define gradient-descent-based optimizers."""

from copy import deepcopy
from typing import Any, Dict, List, Optional

from declearn.model.api import Model, Vector
from declearn.optimizer.modules import OptiModule
from declearn.typing import Batch


__all__ = [
    "Optimizer",
]


class Optimizer:
    """Base class to define gradient-descent-based optimizers.

    The `Optimizer` class defines an API that is required by other
    declearn components for federated learning processes to run.
    It is also fully-workable and is designed to be customizable
    through the use of "plug-in modules" rather than subclassing
    (which might be used for advanced algorithm modifications):
    see `declearn.optimizer.modules.OptiModule` for API details.

    The process implemented here is the following:
    * Compute or receive the (pseudo-)gradients of a model.
    * Refine those by running them through the plug-in modules,
      which are thus composed by sequential application.
    * Optionally compute a decoupled weight decay term (see [1])
      and add it to the updates (i.e. refined gradients).
    * Apply the learning rate and perform the weights' udpate.

    Most plug-in modules are self-contained, in the sense that they
    do not require any information flow between the server and its
    clients in a federated process, and may be used solely by the
    server, by clients or even by a subset of clients - at least
    formally (their might be correctness or convergence issues with
    clients not adopting similar local optimization strategies).

    However, some algorithms designed (or adapted) specifically for
    federated learning require some form of synchronicity between
    the server and clients. In that case, they should be coded to
    emit and expect auxiliary variables, shared between server and
    clients alongside updated model weights during training. Those
    mechanisms are to be implemented at the level of the modules
    themselves, but are wrapped at optimizer level, which collects
    plugged-in-modules' variables and maps back received variables
    to them.

    Attributes
    ----------

    lrate: float
        Base learning rate applied to computed updates.
    w_decay: float
        Decoupled weight decay parameter.
    modules: list[OptiModule]
        List of plug-in modules composed into the optimizer's
        gradients-to-updates computation algorithm.

    API methods:
    -----------
    apply_gradients(Model, Vector) -> None:
        Update a Model based on a pre-computed Vector of gradients.
    collect_aux_var() -> Dict[str, Dict[str, Any]]:
        Collect and package plug-in modules' auxiliary variables.
    process_aux_var(Dict[str, Dict[str, Any]]) -> None:
        Pass auxiliary variables to plug-in modules for processing.
    run_train_step(Model, batch) -> None:
        Compute gradients of a Model over a Batch and apply updates.

    References
    ----------
    [1] Loshchilov & Hutter, 2019.
        Decoupled Weight Decay Regularization.
        https://arxiv.org/abs/1711.05101
    """

    def __init__(
        self,
        lrate: float,  # future: add scheduling tools
        w_decay: float = 0.0,  # future: add scheduling tools
        modules: Optional[List[OptiModule]] = None,
    ) -> None:
        """Instantiate the gradient-descent optimizer.

        Parameters
        ----------
        lrate: float
            Base learning rate (i.e. step size) applied to gradients-
            based updates upon applying them to a model's weights.
        w_decay: float, default=0.
            Optional weight decay parameter, used to parameterize
            a decoupled weight decay regularization term (see [1])
            added to the updates right before the learning rate is
            applied and model weights are effectively updated.
        modules: list[OptiModule] or None, default=None
            Optional list of plug-in modules implementing gradients'
            alteration into model weights' udpates. Modules will be
            applied to gradients following this list's ordering.
            See `declearn.optimizer.modules.OptiModule` for details.

        References
        ----------
        [1] Loshchilov & Hutter, 2019.
            Decoupled Weight Decay Regularization.
            https://arxiv.org/abs/1711.05101
        """
        self.lrate = lrate
        self.w_decay = w_decay
        self.modules = [] if modules is None else modules
        for module in self.modules:
            if not isinstance(module, OptiModule):
                raise TypeError(
                    "'modules' should be a list of `OptiModule` instances; "
                    f"received an element of type '{type(module).__name__}'."
                )

    def get_config(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict with this optimizer's parameters."""
        return {
            "lrate": self.lrate,
            "w_decay": self.w_decay,
            "modules": [mod.serialize().to_dict() for mod in self.modules],
        }

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> "Optimizer":
        """Instantiate an Optimizer from its configuration dict."""
        config = deepcopy(config)  # avoid side-effects
        config["modules"] = [
            OptiModule.deserialize(cfg) for cfg in config.pop("modules", [])
        ]
        return cls(**config)

    def apply_gradients(
        self,
        model: Model,
        gradients: Vector,
    ) -> None:
        """Compute and apply model updates based on pre-computed gradients.

        Parameters
        ----------
        model: Model
            Model instance that is to be trained using gradient-descent.
        gradients: Vector
            Pre-computed vector of (pseudo-)gradients based on which to
            perform the gradient-descent step, by applying the algorithm
            defined by this optimizer and its plug-in modules.

        Returns
        -------
        None
            This method does not return, as `model` is updated in-place.
        """
        # Run input gradients through plug-in modules.
        for module in self.modules:
            gradients = module.run(gradients)
        # Apply the base learning rate.
        updates = self.lrate * gradients
        # Optionally add the decoupled weight decay term.
        if self.w_decay:
            updates += self.w_decay * model.get_weights()
        # Apply updates to the model.
        model.apply_updates(-1.0 * updates)

    def collect_aux_var(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """Return auxiliary variables that need to be shared between nodes.

        Returns
        -------
        aux_var: dict[str, dict[str, ...]]
            Dict that associates `module.collect_aux_var()` values
            to `module.name` keys for each and every module plugged
            in this optimizer that does produce auxiliary variables.
        """
        aux_var = {}  # type: Dict[str, Dict[str, Any]]
        for module in self.modules:
            auxv = module.collect_aux_var()
            if auxv:
                aux_var[module.name] = auxv
        return aux_var

    def process_aux_var(
        self,
        aux_var: Dict[str, Dict[str, Any]],
    ) -> None:
        """Update plug-in modules based on received shared auxiliary variables.

        Received auxiliary variables will be passed to this optimizer's
        modules' `process_aux_var` method, mapped based on `module.name`.

        Parameters
        ----------
        aux_var: dict[str, dict[str, ...]]
            Auxiliary variables received from the counterpart optimizer
            (on the other side of the client/server relationship), that
            are to be a {`module.name`: `module.collect_aux_var()`} *or*
            a {`module.name`: {client: `module.collect_aux_var()`}} dict
            (depending on which side this optimizer is on).

        Raises
        ------
        KeyError
            If a key from `aux_var` does not match the name of any module
            plugged in this optimizer (i.e. if received variables cannot
            be mapped to a destinatory module).
        """
        modules = {module.name: module for module in self.modules}
        for name, auxv in aux_var.items():
            module = modules.get(name)
            if module is None:
                raise KeyError(
                    f"No module with name '{name}' is available to receive "
                    "auxiliary variables."
                )
            module.process_aux_var(auxv)

    def run_train_step(
        self,
        model: Model,
        batch: Batch,
    ) -> None:
        """Perform a gradient-descent step on a given batch.

        Parameters
        ----------
        model: Model
            Model instance that is to be trained using gradient-descent.
        batch: Batch
            Training data used for that training step.

        Returns
        -------
        None
            This method does not return, as `model` is updated in-place.
        """
        gradients = model.compute_batch_gradients(batch)
        self.apply_gradients(model, gradients)
