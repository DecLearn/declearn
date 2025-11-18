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

"""SCAFFOLD algorithm for FL, implemented as a pair of plug-in modules.

The pair of `OptiModule` classes implemented here serve to implement
the SCAFFOLD (Stochastic Averaging for Federated Learning) algorithm
as a plug-in option to optimizer processes.

They only implement Option-II of the paper regarding client-specific
state variables' update, and implementing Option-I would require the
use of a specific Optimizer sub-class.

References
----------
[1] Karimireddy et al., 2019.
    SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.
    https://arxiv.org/abs/1910.06378
"""

import dataclasses
import uuid
import warnings
from typing import Any, Dict, Optional, Set, Tuple, Union

from declearn.model.api import Vector
from declearn.optimizer.modules._api import AuxVar, OptiModule

__all__ = [
    "ScaffoldAuxVar",
    "ScaffoldClientModule",
    "ScaffoldServerModule",
]


@dataclasses.dataclass
class ScaffoldAuxVar(AuxVar):
    """AuxVar subclass for Scaffold.

    - In Server -> Client direction, `state` should be specified.
    - In Client -> Server direction, `delta` should be specified.
    """

    state: Union[Vector, float, None] = None
    delta: Union[Vector, float, None] = None
    clients: Set[str] = dataclasses.field(default_factory=set)

    def __post_init__(
        self,
    ) -> None:
        if ((self.state is None) + (self.delta is None)) != 1:
            raise ValueError(
                "'ScaffoldAuxVar' should have exactly one of 'state' or "
                "'delta' specified as a Vector or conventional 0.0 value."
            )
        if isinstance(self.clients, list):
            self.clients = set(self.clients)

    @staticmethod
    def aggregate_clients(
        val_a: Set[str],
        val_b: Set[str],
    ) -> Set[str]:
        """Custom aggregation rule for 'clients' field."""
        return val_a.union(val_b)

    @classmethod
    def aggregate_state(
        cls,
        val_a: Union[Vector, float, None],
        val_b: Union[Vector, float, None],
    ) -> None:
        """Custom aggregation rule for 'state' field, raising when due."""
        if (val_a is not None) or (val_b is not None):
            raise NotImplementedError(
                "'ScaffoldAuxVar' should not be aggregating 'state' values."
            )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        if self.state is not None:
            output["state"] = self.state
        if self.delta is not None:
            output["delta"] = self.delta
        if self.clients:
            output["clients"] = list(self.clients)
        return output

    def prepare_for_secagg(
        self,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        if self.state is not None:
            raise NotImplementedError(
                "'ScaffoldAuxVar' with 'state' should not undergo SecAgg."
            )
        return {"delta": self.delta}, {"clients": self.clients}


class ScaffoldClientModule(OptiModule[ScaffoldAuxVar]):
    """Client-side Stochastic Controlled Averaging (SCAFFOLD) module.

    This module is to be added to the optimizer used by a federated-
    learning client, and expects that the server's optimizer use its
    counterpart module:
    [`ScaffoldServerModule`][declearn.optimizer.modules.ScaffoldServerModule].

    This module implements the following algorithm:

        Init:
            state = 0
            local = 0
            delta = 0
            _past = 0
            _step = 0
        Step(grads):
            _past += grads
            _step += 1
            grads = grads - delta
        Send -> l_upd:
            loc_n = (_past / _step)
            l_upd = loc_n - local
            local = loc_n
        Receive(state):
            state = state
            delta = local - state
            reset(_past, _step) to 0

    In other words, this module applies a correction term to each
    and every input gradient, which is defined as the difference
    between a local (node-specific) state and a global one, which
    is received from a paired server-side module. At the end of a
    training round (made of multiple steps) it computes an updated
    local state based on the accumulated sum of raw input gradients.
    The difference between the new and previous local states is then
    shared with the server, that aggregates client-wise updates into
    the new global state and emits it towards nodes in return.

    The SCAFFOLD algorithm is described in reference [1].
    The server-side correction of aggregated gradients, the storage
    of raw local and shared states, and the computation of the updated
    shared state and derived client-wise delta values are deferred to
    `ScaffoldServerModule`.

    The formula applied to compute the updated local state variables
    corresponds to the "Option-II" in the paper.
    Implementing Option-I would require holding a copy of the shared
    model and computing its gradients in addition to those of the
    local model, effectively doubling computations. This can be done
    in `declearn`, but requires implementing an alternative training
    procedure rather than an optimizer plug-in.

    References
    ----------
    [1] Karimireddy et al., 2019.
        SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.
        https://arxiv.org/abs/1910.06378
    """

    name = "scaffold-client"
    aux_name = "scaffold"

    def __init__(
        self,
    ) -> None:
        """Instantiate the client-side SCAFFOLD gradients-correction module."""
        self.uuid = str(uuid.uuid4())
        self.state: Union[Vector, float] = 0.0
        self.delta: Union[Vector, float] = 0.0
        self.sglob: Union[Vector, float] = 0.0
        self._grads: Union[Vector, float] = 0.0
        self._steps = 0

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        # Accumulate the uncorrected gradients.
        self._grads = self._grads + gradients
        self._steps += 1
        # Apply state-based correction to outputs.
        return gradients - self.delta

    def collect_aux_var(
        self,
    ) -> Optional[ScaffoldAuxVar]:
        """Return auxiliary variables that need to be shared between nodes.

        Compute and package (without applying it) the updated value
        of the local state variable, so that the server may compute
        the updated shared state variable.

        Returns
        -------
        aux_var:
            Auxiliary variables that are to be shared, aggregated and
            eventually passed to a server-held `ScaffoldServerModule`.

        Warns
        -----
        RuntimeWarning
            If called on an instance that has not processed any gradients
            (via a call to `run`) since the last call to `process_aux_var`
            (or its instantiation).
        """
        # Warn and return an empty dict if no steps were run.
        if not self._steps:
            warnings.warn(
                "Collecting auxiliary variables from a scaffold module "
                "that was not run. The local state update was skipped, "
                "and empty auxiliary variables are emitted.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        # Compute the updated local state and assign it.
        state_next = self._compute_updated_state()
        state_updt = state_next - self.state
        self.state = state_next
        # Send the local state's update.
        return ScaffoldAuxVar(delta=state_updt, clients={self.uuid})

    def _compute_updated_state(
        self,
    ) -> Vector:
        """Compute and return the updated value of the local state.

        Note: the computed update is *not* applied by this method.

        The computation implemented here is equivalent to "Option II"
        of the SCAFFOLD paper. In that paper, authors write that:
            c_i^+ = (c_i - c) + (x - y_i) / (K * eta_l)
        where x are the shared model's weights, y_i are the local
        model's weights after K optimization steps with eta_l lr,
        c is the shared global state and c_i is the local state.

        Noting that (x - y_i) is in fact the difference between the
        local model's weights before and after running K training
        steps, we rewrite it as eta_l * Sum_k(grad(y_i^k) - D_i),
        where we define D_i = (c_i - c). Thus we rewrite c_i^+ as:
            c_i^+ = D_i + (1/K)*Sum_k(grad(y_i^k) - D_i)
        Noting that D_i is constant across steps, we take it out of
        the summation term, leaving us with:
            c_i^+ = (1/K)*Sum_k(grad(y_i^k))

        Hence the new local state can be computed by averaging the
        gradients input to this module along the training steps.
        """
        if not self._steps:  # pragma: no cover
            raise ValueError(
                "Cannot compute an updated state when no steps were run."
            )
        if not isinstance(self._grads, Vector):  # pragma: no cover
            raise TypeError(
                "Internal gradients accumulator is not a Vector instance. "
                "This seems to indicate that the Scaffold module received "
                "improper-type inputs, which should not be possible."
            )
        return self._grads / self._steps

    def process_aux_var(
        self,
        aux_var: ScaffoldAuxVar,
    ) -> None:
        """Update this module based on received shared auxiliary variables.

        Collect the (local_state - shared_state) variable sent by server.
        Reset hidden variables used to compute the local state's updates.

        Parameters
        ----------
        aux_var:
            Auxiliary variables that are to be processed by this module,
            emitted by a counterpart OptiModule on the other side of the
            client-server relationship.

        Raises
        ------
        KeyError
            If `aux_var` is empty.
        TypeError
            If `aux_var` has unproper type.
        """
        if not isinstance(aux_var, ScaffoldAuxVar):
            raise TypeError(
                f"'{self.__class__.__name__}.process_aux_var' received "
                f"auxiliary variables of unproper type: '{type(aux_var)}'."
            )
        if aux_var.state is None:
            raise KeyError(
                "Missing 'state' data in auxiliary variables passed to "
                f"'{self.__class__.__name__}.process_aux_var'."
            )
        # Assign new global state and update the local correction term.
        self.sglob = aux_var.state
        self.delta = self.state - self.sglob
        # Reset internal local variables.
        self._grads = 0.0
        self._steps = 0

    def get_state(
        self,
    ) -> Dict[str, Any]:
        return {
            "state": self.state,
            "sglob": self.sglob,
            "uuid": self.uuid,
        }

    def set_state(
        self,
        state: Dict[str, Any],
    ) -> None:
        for key in ("state", "sglob", "uuid"):
            if key not in state:
                raise KeyError(f"Missing required state variable '{key}'.")
        # Assign received information.
        self.state = state["state"]
        self.sglob = state["sglob"]
        self.uuid = state["uuid"]
        # Reset correction state and internal local variables.
        self.delta = self.state - self.sglob
        self._grads = 0.0
        self._steps = 0


class ScaffoldServerModule(OptiModule[ScaffoldAuxVar]):
    """Server-side Stochastic Controlled Averaging (SCAFFOLD) module.

    This module is to be added to the optimizer used by a federated-
    learning server, and expects that the clients' optimizer use its
    counterpart module:
    [`ScaffoldClientModule`][declearn.optimizer.modules.ScaffoldClientModule].

    This module implements the following algorithm:

        Init:
            s_state = 0
            clients = {}
        Step(grads):
            grads
        Send -> state:
            state = s_state / min(len(clients), 1)
        Receive(delta=sum(state_i^t+1 - state_i^t), clients=set{uuid}):
            s_state += delta
            clients.update(clients)

    In other words, this module holds a global state variable, set
    to zero at instantiation. At the beginning of a training round
    it sends it to all clients so that they can derive a correction
    term for their processed gradients, based on a local state they
    hold. At the end of a training round, client-wise local state
    updates are sum-aggregated into an update for the global state
    variable, which will be sent to clients at the start of the next
    round. The sent state is always the average of the last known
    local state from each and every known client.

    The SCAFFOLD algorithm is described in reference [1].
    The client-side correction of gradients and the computation of
    updated local states are deferred to `ScaffoldClientModule`.

    References
    ----------
    [1] Karimireddy et al., 2019.
        SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.
        https://arxiv.org/abs/1910.06378
    """

    name = "scaffold-server"
    aux_name = "scaffold"
    auxvar_cls = ScaffoldAuxVar

    def __init__(
        self,
    ) -> None:
        """Instantiate the server-side SCAFFOLD gradients-correction module."""
        self.s_state: Union[Vector, float] = 0.0
        self.clients: Set[str] = set()

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        # Note: ScaffoldServer only manages auxiliary variables.
        return gradients

    def collect_aux_var(
        self,
    ) -> ScaffoldAuxVar:
        """Return auxiliary variables that need to be shared between nodes.

        Returns
        -------
        aux_var:
            `ScaffoldAuxVar` instance holding auxiliary variables that are
            to be shared with clients' `ScaffoldClientModule` instances.
        """
        # When un-initialized, send lightweight information.
        if not self.clients:
            return ScaffoldAuxVar(state=0.0)
        # Otherwise, compute and return the current shared state.
        return ScaffoldAuxVar(state=self.s_state / len(self.clients))

    def process_aux_var(
        self,
        aux_var: ScaffoldAuxVar,
    ) -> None:
        """Update this module based on received shared auxiliary variables.

        Update the global state variable based on the sum of client's
        local state updates.

        Parameters
        ----------
        aux_var:
            `ScaffoldAuxVar` resulting from the aggregation of clients'
            `ScaffoldClientModule` auxiliary variables.

        Raises
        ------
        KeyError:
            If `aux_var` is empty.
        TypeError:
            If `aux_var` is of unproper type.
        """
        if not isinstance(aux_var, ScaffoldAuxVar):
            raise TypeError(
                f"'{self.__class__.__name__}.process_aux_var' received "
                f"auxiliary variables of unproper type: '{type(aux_var)}'."
            )
        if aux_var.delta is None:
            raise KeyError(
                f"'{self.__class__.__name__}.process_aux_var' received "
                "auxiliary variables with an empty 'delta' field."
            )
        # Update the list of known clients, and the sum of local states.
        self.clients.update(aux_var.clients)
        self.s_state += aux_var.delta

    def get_state(
        self,
    ) -> Dict[str, Any]:
        return {"s_state": self.s_state, "clients": list(self.clients)}

    def set_state(
        self,
        state: Dict[str, Any],
    ) -> None:
        for key in ("s_state", "clients"):
            if key not in state:
                raise KeyError(f"Missing required state variable '{key}'.")
        self.s_state = state["s_state"]
        self.clients = set(state["clients"])
