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

"""Base class to define gradient-descent-based optimizers."""

from typing import (  # fmt: off
    Any,
    Dict,
    List,
    Optional,
    Self,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from declearn.model.api import Model, Vector
from declearn.optimizer.modules import AuxVar, OptiModule
from declearn.optimizer.regularizers import Regularizer
from declearn.optimizer.schedulers import Scheduler
from declearn.typing import Batch

__all__ = [
    "Optimizer",
]


T = TypeVar("T")


class ConstantScheduler(Scheduler, register=False):
    """Ad-hoc Scheduler subclass for constant values."""

    def get_next_rate(
        self,
    ) -> float:
        return self.base

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        return self.base


class Optimizer:
    """Base class to define gradient-descent-based optimizers.

    The `Optimizer` class defines an API that is required by other
    declearn components for federated learning processes to run.
    It is also fully-workable and is designed to be customizable
    through the use of "plug-in modules" rather than subclassing
    (which might be used for advanced algorithm modifications):
    see the base classes [declearn.optimizer.modules.OptiModule][]
    and [declearn.optimizer.regularizers.Regularizer][] for details.

    The process implemented here is the following:

    - Compute or receive the (pseudo-)gradients of a model.
    - Compute loss-regularization terms and add them to the
      gradients, based on a list of plug-in regularizers.
    - Refine gradients by running them through plug-in modules,
      which are thus composed by sequential application.
    - Optionally compute a decoupled weight decay term (see [1])
      and add it to the updates (i.e. refined gradients).
    - Apply the learning rate and perform the weights' udpate.

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

    Finally, both the learning rate and weight decay rate may be
    specified as a [declearn.optimizer.schedulers.Scheduler][]
    instance, implementing a time-based rule to have the value
    evolve throughout training.

    Attributes
    ----------
    lrate: float
        Base learning rate that will be applied next time updates
        are computed from gradients. Read-only property.
    w_decay: float
        Decoupled weight decay parameter that will be applied next
        time updates are computed from gradients. Read-only property.
    modules: list[OptiModule]
        List of plug-in modules composed into the optimizer's
        gradients-to-updates computation algorithm.
    regularizers: list[Regularizer]
        List of plug-in loss regularization modules composed into
        the optimizer's gradients-to-updates computation algorithm.

    API methods
    -----------
    - apply_gradients(Model, Vector) -> None:
        Update a Model based on a pre-computed Vector of gradients.
    - collect_aux_var() -> Dict[str, AuxVar]:
        Collect and package plug-in modules' auxiliary variables.
    - compute_updates_from_gradients(Model, Vector) -> Vector:
        Compute and return model updates based on pre-computed gradients.
    - process_aux_var(Dict[str, AuxVar]) -> None:
        Pass auxiliary variables to plug-in modules for processing.
    - run_train_step(Model, batch) -> None:
        Compute gradients of a Model over a Batch and apply updates.
    - start_round() -> None:
        Signal that a new training round is starting to wrapped regularizers.

    References
    ----------
    [1] Loshchilov & Hutter, 2019.
        Decoupled Weight Decay Regularization.
        https://arxiv.org/abs/1711.05101

    See also
    --------
    - [declearn.optimizer.list_optim_modules][]:
        Return a mapping of registered OptiModule subclasses.
    - [declearn.optimizer.list_optim_regularizers][]:
        Return a mapping of registered Regularizer subclasses.
    - [declearn.optimizer.list_rate_schedulers][]:
        Return a mapping of registered Scheduler subclasses.
    """

    def __init__(
        self,
        lrate: Union[float, Scheduler, Tuple[str, Dict[str, Any]]],
        w_decay: Union[float, Scheduler, Tuple[str, Dict[str, Any]]] = 0.0,
        regularizers: Optional[
            Sequence[Union[Regularizer, str, Tuple[str, Dict[str, Any]]]]
        ] = None,
        modules: Optional[
            Sequence[Union[OptiModule, str, Tuple[str, Dict[str, Any]]]]
        ] = None,
    ) -> None:
        """Instantiate the gradient-descent optimizer.

        Parameters
        ----------
        lrate: float or Scheduler or specs
            Base learning rate (i.e. step size) applied to gradients-
            based updates upon applying them to a model's weights.
            May be a constant value, or a `Scheduler` instance or specs.
            See `declearn.optimizer.schedulers.Scheduler` for details.
        w_decay: float or Scheduler or specs, default=0.
            Optional weight decay parameter, used to parameterize
            a decoupled weight decay regularization term (see [1])
            added to the updates right before the learning rate is
            applied and model weights are effectively updated.
            May be a constant value, or a `Scheduler` instance or specs.
        regularizers: list[Regularizer or specs] or None, default=None
            Optional list of plug-in loss regularizers. Regularizers will
            be applied to gradients following this list's order, prior to
            any other alteration (e.g. accelaration module - see below).
            See `declearn.optimizer.regularizers.Regularizer` for details.
            See Notes section below for details on the "specs" format.
        modules: list[OptiModule or specs] or None, default=None
            Optional list of plug-in modules implementing gradients'
            alteration into model weights' udpates. Modules will be
            applied to gradients following this list's ordering.
            See `declearn.optimizer.modules.OptiModule` for details.
            See Notes section below for details on the "specs" format.

        Notes
        -----

        * `Regularizer` and `OptiModule` to be used by this optimizer,
          specified using the `regularizers` and `modules` parameters,
          may be passed as ready-for-use instances, or be instantiated
          from specs, consisting either of a single string (the `name`
          attribute of the class to build) or a tuple grouping this
          name and a config dict (to specify some hyper-parameters).
        * `Scheduler` instances to be used by this optimizer to adjust
          the `lrate` and/or `w_decay` parameters through time may be
          passed as ready-for-use instances, or be instantiated from
          specs, consisting of a tuple grouping a string (the `name`
          attribute of the class to build) and and a config dict (with
          the base value and algorithm-specific hyper-parameters).

        References
        ----------
        [1] Loshchilov & Hutter, 2019.
            Decoupled Weight Decay Regularization.
            https://arxiv.org/abs/1711.05101
        """
        self._lrate_scheduler = self._parse_scheduler(lrate)
        self._wrate_scheduler = self._parse_scheduler(w_decay)
        self._lrate = self._lrate_scheduler.get_next_rate()
        self._wrate = self._wrate_scheduler.get_next_rate()
        self.regularizers: List[Regularizer] = (
            []
            if regularizers is None
            else self._parse_plugins(Regularizer, regularizers)  # type: ignore
        )
        self.modules: List[OptiModule] = (
            [] if modules is None else self._parse_plugins(OptiModule, modules)  # type: ignore
        )

    @property
    def lrate(self) -> float:
        """Current learning rate for this Optimizer."""
        return self._lrate

    @property
    def w_decay(self) -> float:
        """Current weight decay rate for this Optimizer."""
        return self._wrate

    def _update_rates(
        self,
    ) -> None:
        """Update the learning and weight decay rates as scheduled."""
        self._lrate = self._lrate_scheduler.get_next_rate()
        self._wrate = self._wrate_scheduler.get_next_rate()

    def _parse_scheduler(
        self,
        value: Union[float, Scheduler, Tuple[str, Dict[str, Any]]],
    ) -> Scheduler:
        if isinstance(value, float):
            return ConstantScheduler(base=value)
        if isinstance(value, Scheduler):
            return value
        if isinstance(value, (tuple, list)) and (len(value) == 2):
            name, config = value
            return Scheduler.from_specs(name, config)
        raise TypeError(
            f"Cannot instantiate a 'Scheduler' from {value}. "
            "Required a float, 'Scheduler' or specs ((str, dict) tuple)."
        )

    def _parse_plugins(
        self,
        cls: Type[Union[OptiModule, Regularizer]],
        plugins: Sequence[Union[Any, str, Tuple[str, Dict[str, Any]]]],
    ) -> Union[List[OptiModule], List[Regularizer]]:
        """Parse a list of plug-in specs into a list of instances.

        Parameters
        ----------
        cls: Type[OptiModule or Regularizer]
            Base type of plug-ins being instantiated.
        plugins: list[`cls` | str | (str, dict)]
            List of instances or specifications to process and/or type-check.
            Specifications may be a single string (`name` attribute of the
            type to build) or a tuple grouping this name and a config dict
            (to specify non-default hyper-parameters).

        Returns
        -------
        plugins: list[`cls`]
            List of `cls` instances created (or taken) from the specs.
        """
        output = []
        for specs in plugins:
            if isinstance(specs, cls):
                plugin = specs
            elif isinstance(specs, str):
                plugin = cls.from_specs(specs, config={})
            elif isinstance(specs, (tuple, list)) and (len(specs) == 2):
                plugin = cls.from_specs(*specs)
            else:
                raise TypeError(
                    f"Cannot instantiate a {cls.__name__} from {specs}. "
                    "Required a name (str) or specs ((str, dict) tuple)."
                )
            output.append(plugin)
        return output  # type: ignore

    def get_config(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict with this optimizer's parameters.

        The counterpart to this method is the `from_config` classmethod.
        To access the optimizer's inner states, see the `get_state` method.

        Returns
        -------
        config: dict[str, any]
            JSON-serializable dict storing this optimizer's instantiation
            configuration.
        """

        lrate = (
            (self._lrate_scheduler.name, self._lrate_scheduler.get_config())
            if not isinstance(self._lrate_scheduler, ConstantScheduler)
            else self._lrate_scheduler.base
        )
        decay = (
            (self._wrate_scheduler.name, self._wrate_scheduler.get_config())
            if not isinstance(self._wrate_scheduler, ConstantScheduler)
            else self._wrate_scheduler.base
        )
        regulzr = [(reg.name, reg.get_config()) for reg in self.regularizers]
        modules = [(mod.name, mod.get_config()) for mod in self.modules]
        return {
            "lrate": lrate,
            "w_decay": decay,
            "regularizers": regulzr,
            "modules": modules,
        }

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        """Instantiate an Optimizer from its configuration dict.

        The counterpart to this classmethod is the `get_config` method.
        To restore the optimizer's inner states, see its `get_state` method.

        Parameters
        ----------
        config: dict[str, Any]
            Dict storing the optimizer's instantiation configuration.

        Raises
        ------
        KeyError
            If the provided `config` lacks some required parameters
            and/or contains some unused ones.
        """
        return cls(**config)

    def compute_updates_from_gradients(
        self,
        model: Model[Vector[T]],
        gradients: Vector[T],
    ) -> Vector[T]:
        """Compute and return model updates based on pre-computed gradients.

        Parameters
        ----------
        model: Model
            Model instance that is to be trained using gradient-descent.
            This parameter is only used to access current weights in case
            some loss regularizers are part of the pipeline.
        gradients: Vector
            Pre-computed vector of (pseudo-)gradients based on which to
            perform the gradient-descent step, by applying the algorithm
            defined by this optimizer and its plug-in modules.

        Returns
        -------
        updates: Vector
            Model weights' updates, preserving input `gradients`'s specs,
            ready to be applied using the `model.apply_updates` method.
        """
        # Optionally fetch the model's trainable weights.
        if self.regularizers or self.w_decay:
            weights = model.get_weights(trainable=True)
        # Run input gradients and weights through plug-in regularizers.
        if self.regularizers:
            # false-positive; pylint: disable=possibly-used-before-assignment
            for regularizer in self.regularizers:
                gradients = regularizer.run(gradients, weights)
        # Run input gradients through plug-in modules.
        for module in self.modules:
            gradients = module.run(gradients)
        # Apply the base learning rate.
        updates = self.lrate * gradients
        # Optionally add the decoupled weight decay term.
        if self.w_decay:
            updates += self.w_decay * weights
        # Update the learning and weight decay rates.
        self._update_rates()
        # Return ready-to-apply model updates.
        return -1.0 * updates

    def collect_aux_var(
        self,
    ) -> Dict[str, AuxVar]:
        """Return auxiliary variables that need to be shared between nodes.

        Returns
        -------
        aux_var:
            Dict that associates `module.collect_aux_var()` values
            to `module.name` keys for each and every module plugged
            in this optimizer that produces auxiliary variables.
        """
        aux_var: Dict[str, AuxVar] = {}
        for module in self.modules:
            auxv = module.collect_aux_var()
            if auxv is not None:
                name = module.aux_name or module.name
                aux_var[name] = auxv
        return aux_var

    def process_aux_var(
        self,
        aux_var: Dict[str, AuxVar],
    ) -> None:
        """Update plug-in modules based on received shared auxiliary variables.

        Received auxiliary variables will be passed to this optimizer's
        modules' `process_aux_var` method, mapped based on `module.name`.

        Parameters
        ----------
        aux_var: dict[str, AuxVar]
            Auxiliary variables received from the counterpart optimizer
            (on the other side of the client/server relationship), that
            are packed as a `{module.name: module.auxvar_cls}` dict for
            modules that do use auxiliary variables.
            When auxiliary variables from multiple peers are due to be
            processed, they must be aggregated prior to being passed to
            this method.

        Raises
        ------
        KeyError
            If a key from `aux_var` does not match the name of any module
            plugged in this optimizer (i.e. if received variables cannot
            be mapped to a destinatory module).
        """
        modules = {
            (module.aux_name or module.name): module for module in self.modules
        }
        for name, auxv in aux_var.items():
            module = modules.get(name, None)
            if module is None:
                raise KeyError(
                    f"No module with name '{name}' is available to receive "
                    "auxiliary variables."
                )
            module.process_aux_var(auxv)

    def start_round(
        self,
    ) -> None:
        """Perform any required action at the start of a training round.

        This method calls the `on_round_start` callback of each and every
        wrapped `Regularizer` which may be used to regulate some internal
        state variables, as well as that of the `Scheduler` objects that
        regulate the evolution of the learning and weight decay rates.
        """
        self._lrate_scheduler.on_round_start()
        self._wrate_scheduler.on_round_start()
        for regularizer in self.regularizers:
            regularizer.on_round_start()

    def run_train_step(
        self,
        model: Model,
        batch: Batch,
        sclip: Optional[float] = None,
    ) -> None:
        """Perform a gradient-descent step on a given batch.

        Parameters
        ----------
        model: Model
            Model instance that is to be trained using gradient-descent.
        batch: Batch
            Training data used for that training step.
        sclip: float or None, default=None
            Optional L2-norm clipping threshold for sample-wise gradients,
            restraining their sensitivity prior to any alteration designed
            as part of this Optimizer's pipeline of plug-in algorithms.

        Returns
        -------
        None
            This method does not return, as `model` is updated in-place.
        """
        gradients = model.compute_batch_gradients(batch, max_norm=sclip)
        self.apply_gradients(model, gradients)

    def apply_gradients(
        self,
        model: Model[Vector[T]],
        gradients: Vector[T],
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
        updates = self.compute_updates_from_gradients(model, gradients)
        model.apply_updates(updates)

    def get_state(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict with this optimizer's state.

        The counterpart to this method is the `set_state` one.

        Returns
        -------
        state: dict[str, any]
            JSON-serializable dict storing this optimizer's inner state
            variables (i.e. those from its modules).
        """
        lrate = self._lrate_scheduler.get_state()
        wrate = self._wrate_scheduler.get_state()
        modules = [(mod.name, mod.get_state()) for mod in self.modules]
        return {"modules": modules, "lrate": lrate, "w_decay": wrate}

    def set_state(
        self,
        states: Dict[str, Any],
    ) -> None:
        """Load a saved state dict into an optimizer instance.

        The counterpart to this method is the `get_state` one.

        Parameters
        ----------
        states: dict[str, any]
            Dict storing values to assign to this optimizer's inner
            state variables (i.e. those from its modules).

        Raises
        ------
        KeyError
            If the received states do not match the expected config,
            whether because a module is missing or one of its states
            is missing.
            In both cases, the Optimizer's states will be reverted
            to their values prior to the failed call to this method.
        RuntimeError
            If a KeyError was raised both when trying to apply the
            input `state` and when trying to revert the states to
            their initial values after that first error was raised.
            This should never happen and indicates a source code
            error in a wrapped module, or even in this class.
        """
        for key in ("lrate", "w_decay", "modules"):
            if key not in states:
                raise KeyError(
                    f"Optimizer input 'states' lack a '{key}' field."
                )
        if len(states["modules"]) != len(self.modules):
            raise KeyError("Optimizer 'states' do not match modules config.")
        initial = self.get_state()
        try:
            self._set_state(states)
        except KeyError as exc:
            try:
                self._set_state(initial)
            except KeyError as exc_bis:
                raise RuntimeError(
                    "`Optimizer.set_state` failed to restore initial states "
                    "after a KeyError was raised during states' attempted "
                    "update. There probably is a source code error with one "
                    "of the wrapped modules.\n"
                    f"Error when reverting states: {exc_bis}\n"
                    f"Initial update error: {exc}\n"
                ) from exc_bis
            raise exc

    def _set_state(
        self,
        states: Dict[str, Any],
    ) -> None:
        """Backend to the `set_state` method, lacking exception-catching."""
        self._lrate_scheduler.set_state(states["lrate"])
        self._wrate_scheduler.set_state(states["w_decay"])
        for mod, (name, state) in zip(
            self.modules, states["modules"], strict=False
        ):
            if mod.name != name:
                raise KeyError(
                    "Optimizer 'states' do not match modules config."
                )
            # Note: this may raise a KeyError if 'state' is misspecified.
            mod.set_state(state)
