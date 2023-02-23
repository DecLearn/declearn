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

"""Hacky OptiModule subclass enabling the use of a torch.nn.Optimizer."""

import importlib
from typing import Any, Dict, List, Optional, Union, Tuple, Type

import numpy as np
import torch
from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.model.api import Vector
from declearn.model.torch.utils import select_device
from declearn.model.torch._vector import TorchVector
from declearn.optimizer.modules import OptiModule
from declearn.utils import get_device_policy


__all__ = [
    "TorchOptiModule",
]


class TorchOptiModule(OptiModule):
    """Hacky OptiModule subclass to wrap a torch Optimizer.

    This torch-only OptiModule enables wrapping a `torch.nn.Optimizer`
    to make it part of a declearn Optimizer pipeline, where it may be
    combined with other framework-agnostic tools (notably FL-specific
    ones such as the FedProx loss regularizer).

    The wrapped torch Optimizer states will be placed on a device (CPU
    or GPU) selected automatically based on the first input gradients'
    placement OR on the global device policy when `set_state` is used.
    The `reset` method may be used to drop internal optimizer states
    and device-placement choices at once.

    Please note that this relies on a hack that may have unforeseen side
    effects on the optimization algorithm if used carelessly and will at
    any rate cause some memory overhead. Thus it should be used sparingly,
    taking into account the following constraints and limitations:

    * The wrapped optimizer class should have a "lr" (learning-rate)
      parameter, that will be forced to 1.0, so that updates' scaling
      remains the responsibility of the wrapping declearn Optimizer.
    * The wrapped optimizer class should not make use of the watched
      parameters' values, only of their gradients, because it will in
      fact monitor artificial, zero-valued parameters at each step.
    * If the module is to be used by the clients, the wrapped optimizer
      class must have been imported from a third-party package that is
      also available to the clients (e.g. torch).

    Also note that passing a string input as `optim_cls` (as is always done
    when deserializing the module from its auto-generated config) may raise
    security concerns due to its resulting in importing external code. As a
    consequence, users will be asked to validate any non-torch import before
    it is executed. This may be disabled when instantiating the module from
    its init constructor but not when using `from_config`, `from_specs` or
    `deserialize`.

    This class is mostly provided for experimental use of algorithms that
    are not natively available in declearn, for users that do not want to
    put in (or reserve for later) the effort of writing a custom, dedicated,
    framework-agnostic OptiModule subclass implementing that algorithm.
    If you encounter issues, please report to the declearn developers, and
    we will be happy to assist with debugging the present module and/or
    implementing the desired algorithm as a proper OptiModule.
    """

    name = "torch-optim"

    def __init__(
        self,
        optim_cls: Union[Type[torch.optim.Optimizer], str],
        validate: bool = True,
        **kwargs: Any,
    ) -> None:
        """Instantiate a hacky torch optimizer plug-in module.

        Parameters
        ----------
        optim_cls: type[torch.optim.Optimizer] or str
            Class constructor of the torch Optimizer that needs wrapping.
            A string containing its import path may be provided instead.
        validate: bool, default=True
            Whether the user should be prompted to validate the module-
            import action triggered in case `optim_cls` is a string and
            targets another package than 'torch'.
        **kwargs: Any
            Keyword arguments to `optim_cls`.
            Note that "lr" will be forced to 1.0.
        """
        self.optim_cls = self._validate_optim_cls(optim_cls, validate)
        self.kwargs = kwargs
        self.kwargs["lr"] = 1.0
        self._params = {}  # type: Dict[str, torch.nn.Parameter]
        self._optim = None  # type: Optional[torch.optim.Optimizer]

    def _validate_optim_cls(
        self,
        optim_cls: Union[Type[torch.optim.Optimizer], str],
        validate: bool = True,
    ) -> Type[torch.optim.Optimizer]:
        """Type-check and optionally import a torch Optimizer class.

        Parameters
        ----------
        optim_cls: Type[torch.optim.Optimizer] or str
            Either a torch Optimizer class constructor, or the import path
            to one, from which it will be retrieved.
        validate: bool, default=True
            Whether the user should be prompted to validate the module-
            import action triggered in case `optim_cls` is a string and
            targets another package than 'torch'.

        Raises
        ------
        RuntimeError:
            If `optim_cls` is a string and the target class cannot be loaded.
            If `optim_cls` is a string and the user denies the import command.
        TypeError:
            If `optim_cls` (or the object loaded in case it is a string)
            is not a `torch.nn.Optimizer` subclass.

        Returns
        -------
        optim_cls: Type[torch.optim.Optimizer]
            Torch Optimizer class constructor.
        """
        if isinstance(optim_cls, str):
            try:
                module, name = optim_cls.rsplit(".", 1)
                if validate and (module.split(".", 1)[0] != "torch"):
                    accept = input(
                        f"TorchOptiModule requires importing the '{module}' "
                        "module.\nDo you agree to this? [y/N] "
                    )
                    if not accept.lower().startswith("y"):
                        raise RuntimeError(f"User refused to import {module}.")
                optim_mod = importlib.import_module(module)
                optim_cls = getattr(optim_mod, name)
            except (AttributeError, ModuleNotFoundError, RuntimeError) as exc:
                raise RuntimeError(
                    "Could not load TorchOptiModule's wrapped "
                    f"torch optimizer class: {exc}"
                ) from exc
        if not (
            isinstance(optim_cls, type)
            and issubclass(optim_cls, torch.optim.Optimizer)
        ):
            raise TypeError(
                "'optim_cls' should be a torch Optimizer subclass."
            )
        return optim_cls

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        """Run input gradients through the wrapped torch Optimizer.

        Parameters
        ----------
        gradients: TorchVector
            Input gradients that are to be processed and updated.

        Raises
        ------
        TypeError:
            If `gradients` are not a TorchVector (this module is
            a framework-specific hack).
        KeyError:
            If `gradients` have an inconsistent spec with the first
            ones ever processed by this module. Use `reset` if you
            wish to start back from the beginning.

        Returns
        -------
        gradients: TorchVector
            Modified input gradients. The output Vector should be
            fully compatible with the input one - only the values
            of the wrapped coefficients may have changed.
        """
        if not isinstance(gradients, TorchVector):
            raise TypeError(
                "TorchOptiModule only supports TorchVector input gradients."
            )
        if self._optim is None:
            self._optim = self._init_optimizer(gradients)
        if gradients.coefs.keys() != self._params.keys():
            raise KeyError(
                "Mismatch between input gradients and stored parameters."
            )
        for key, grad in gradients.coefs.items():
            param = self._params[key]
            with torch.no_grad():
                param.zero_()
            param.grad = -grad.to(param.device)  # devices *must* be the same
        self._optim.step()
        coefs = {
            key: param.detach().clone().to(gradients.coefs[key].device)
            for key, param in self._params.items()
        }
        return TorchVector(coefs)

    def _init_optimizer(self, gradients: TorchVector) -> torch.optim.Optimizer:
        """Instantiate and return a torch Optimizer to make use of.

        Place the artifical parameters and optimizer states on the
        same device as the input gradients.
        """
        # false-positive on torch.zeros_like; pylint: disable=no-member
        self._params = {
            key: torch.nn.Parameter(torch.zeros_like(grad))
            for key, grad in gradients.coefs.items()
        }
        return self.optim_cls(list(self._params.values()), **self.kwargs)

    def reset(self) -> None:
        """Reset this module to its uninitialized state.

        Discard the wrapped torch parameters (that define a required
        specification of input gradients) and torch Optimizer. As a
        consequence, the next call to `run` will result in creating
        a new Optimizer from scratch and setting a new specification.
        """
        self._params = {}
        self._optim = None

    def get_config(
        self,
    ) -> Dict[str, Any]:
        optim_cls = f"{self.optim_cls.__module__}.{self.optim_cls.__name__}"
        return {"optim_cls": optim_cls, "kwargs": self.kwargs}

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        if "optim_cls" not in config:
            raise TypeError(
                "TorchOptiModule config is missing required key 'optim_cls'."
            )
        kwargs = config.get("kwargs", {})
        kwargs.pop("validate", None)  # force manual validation of imports
        return cls(config["optim_cls"], validate=True, **kwargs)

    def get_state(
        self,
    ) -> Dict[str, Any]:
        params = TorchVector({k: p.data for k, p in self._params.items()})
        dtypes = params.dtypes()
        shapes = params.shapes()
        specs = {key: (shapes[key], dtypes[key]) for key in self._params}
        sdict = (
            {"state": {}} if self._optim is None else self._optim.state_dict()
        )
        state = []  # type: List[Tuple[int, Dict[str, Any]]]
        for key, group in sdict["state"].items():
            gval = {
                k: v.cpu().numpy().copy() if isinstance(v, torch.Tensor) else v
                for k, v in group.items()
            }
            state.append((key, gval))
        return {"specs": specs, "state": state}

    def set_state(
        self,
        state: Dict[str, Any],
    ) -> None:
        for key in ("specs", "state"):
            if key not in state:
                raise KeyError(
                    "Missing required key in input TorchOptiModule state "
                    f"dict: '{key}'."
                )
        self.reset()
        # Early-exit if reloading from an uninitialized state.
        if not state["state"]:
            return None
        # Consult the global device policy to place the variables and states.
        policy = get_device_policy()
        device = select_device(gpu=policy.gpu, idx=policy.idx)
        # Restore weight variables' specifications from the input state dict.
        self._params = {}
        for key, (shape, dtype) in state["specs"].items():
            zeros = torch.zeros(  # false-positive; pylint: disable=no-member
                tuple(shape), dtype=getattr(torch, dtype), device=device
            )
            self._params[key] = torch.nn.Parameter(zeros)
        self._optim = self.optim_cls(
            list(self._params.values()), **self.kwargs
        )
        # Restore optimizer variables' values from the input state dict.
        sdict = self._optim.state_dict()
        sdict["state"] = {
            key: {
                k: (
                    torch.from_numpy(v).to(device)  # pylint: disable=no-member
                    if isinstance(v, np.ndarray)
                    else v
                )
                for k, v in group.items()
            }
            for key, group in state["state"]
        }
        self._optim.load_state_dict(sdict)
        return None
