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

"""SCAFFOLD algorithm for FL, implemented as a pair of plug-in modules.

The pair of `OptiModule` classes implemented here serve to implement
the SCAFFOLD (Stochastic Averaging for Federated Learning) algorithm
as a plug-in option to optimizer processes.

They only implement Option-II of the paper regarding client-specific
state variables' update, and implementing Option-I would require the
use of a specific Optimizer sub-class.

References:
[1] Karimireddy et al., 2019.
    SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.
    https://arxiv.org/abs/1910.06378
"""

import warnings
from typing import Any, ClassVar, Dict, List, Optional, Union

from declearn.model.api import Vector
from declearn.optimizer.modules._api import OptiModule

__all__ = [
    "ScaffoldClientModule",
    "ScaffoldServerModule",
]


class ScaffoldClientModule(OptiModule):
    """Client-side Stochastic Controlled Averaging (SCAFFOLD) module.

    This module is to be added to the optimizer used by a federated-
    learning client, and expects that the server's optimizer use its
    counterpart module: `ScaffoldServerModule`.

    This module implements the following algorithm:
        Init:
            delta = 0
            _past = 0
            _step = 0
        Step(grads):
            _past += grads
            _step += 1
            grads = grads - delta
        Send:
            state = (_past / _step)
        Receive(delta):
            delta = delta
            reset(_past, _step) to 0

    In other words, this module receives a "delta" variable from the
    server instance which is set as the difference between a client-
    specific state and a shared one, and corrects input gradients by
    adding this delta to it. At the end of a training round (made of
    multiple steps) it computes an updated client state based on the
    accumulated sum of corrected gradients. This value is to be sent
    to the server, that will emit a new value for the local delta in
    return.

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

    References:
    [1] Karimireddy et al., 2019.
        SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.
        https://arxiv.org/abs/1910.06378
    """

    name: ClassVar[str] = "scaffold-client"
    aux_name: ClassVar[str] = "scaffold"

    def __init__(
        self,
    ) -> None:
        """Instantiate the client-side SCAFFOLD gradients-correction module."""
        self.delta = 0.0  # type: Union[Vector, float]
        self._grads = 0.0  # type: Union[Vector, float]
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
    ) -> Optional[Dict[str, Any]]:
        """Return auxiliary variables that need to be shared between nodes.

        Compute and package (without applying it) the updated value
        of the local state variable, so that the server may compute
        the updated shared state variable.
        """
        state = self._compute_updated_state()
        return {"state": state}

    def _compute_updated_state(
        self,
    ) -> Union[Vector, float]:
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
        When then note that D_i is constant and can be taken out
        of the summation term, leaving us with:
            c_i^+ = Avg_k(grad(y_i^k))

        Hence the new local state can be computed by averaging the
        gradients input to this module along the training steps.
        """
        if not self._steps:
            warnings.warn(
                "Collecting auxiliary variables from a scaffold module "
                "that was not run. Returned zero-valued scalar updates.",
                category=RuntimeWarning,
            )
            return 0.0
        return self._grads / self._steps

    def process_aux_var(
        self,
        aux_var: Dict[str, Any],
    ) -> None:
        """Update this module based on received shared auxiliary variables.

        Collect the (local_state - shared_state) variable sent by server.
        Reset hidden variables used to compute the local state's updates.
        """
        # Expect a state variable and apply it.
        delta = aux_var.get("delta", None)
        if delta is None:
            raise KeyError(
                "Missing 'delta' key in ScaffoldClientModule's "
                "received auxiliary variables."
            )
        if isinstance(delta, (float, Vector)):
            self.delta = delta
        else:
            raise TypeError(
                "Unsupported type for ScaffoldClientModule's "
                "received auxiliary variable 'delta'."
            )
        # Reset local variables.
        self._grads = 0.0
        self._steps = 0


class ScaffoldServerModule(OptiModule):
    """Server-side Stochastic Controlled Averaging (SCAFFOLD) module.

    This module is to be added to the optimizer used by a federated-
    learning server, and expects that the clients' optimizer use its
    counterpart module: `ScaffoldClientModule`.

    This module implements the following algorithm:
        Init(clients):
            state = 0
            s_loc = {client: 0 for client in clients}
        Step(grads):
            grads
        Send:
            delta = {client: (s_loc[client] - state); client in s_loc}
        Receive(s_new = {client: state}):
            s_upd = sum(s_new[client] - s_loc[client]; client in s_new)
            s_loc.update(s_new)
            state += s_upd / len(s_loc)

    In other words, this module holds a shared state variable, and a
    set of client-specific ones, which are zero-valued when created.
    At the beginning of a training round it sends to each client its
    delta variable, set to the difference between its current state
    and the shared one, which is to be applied as a correction term
    to local gradients. At the end of a training round, aggregated
    gradients are corrected by substracting the shared state value
    from them. Finally, updated local states received from clients
    are recorded, and used to update the shared state variable, so
    that new delta values can be sent to clients as the next round
    of training starts.

    The SCAFFOLD algorithm is described in reference [1].
    The client-side correction of gradients and the computation of
    updated local states are deferred to `ScaffoldClientModule`.

    References:
    [1] Karimireddy et al., 2019.
        SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.
        https://arxiv.org/abs/1910.06378
    """

    name: ClassVar[str] = "scaffold-server"
    aux_name: ClassVar[str] = "scaffold"

    def __init__(
        self,
        clients: Optional[List[str]] = None,
    ) -> None:
        """Instantiate the server-side SCAFFOLD gradients-correction module.

        Parameters
        ----------
        clients: list[str] or None, default=None
            Optional list of known clients' id strings.

        If this module is used under a training strategy that has
        participating clients vary across epochs, leaving `clients`
        to None will affect the update rule for the shared state,
        as it uses a (n_participating / n_total_clients) term, the
        divisor of which will be incorrect (at least on the first
        step, potentially on following ones as well).
        Similarly, listing clients that in fact do not participate
        in training will have side effects on computations.
        """
        self.state = 0.0  # type: Union[Vector, float]
        self.s_loc = {}  # type: Dict[str, Union[Vector, float]]
        if clients:
            self.s_loc = {client: 0.0 for client in clients}

    def get_config(
        self,
    ) -> Dict[str, Any]:
        return {"clients": list(self.s_loc)}

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        # Note: ScaffoldServer only manages auxiliary variables.
        return gradients

    def collect_aux_var(
        self,
    ) -> Optional[Dict[str, Any]]:
        """Return auxiliary variables that need to be shared between nodes.

        Package client-wise (local_state - shared_state) variables.
        """
        # Compute clients' delta variable, package them and return.
        aux_var = {}  # type: Dict[str, Dict[str, Any]]
        for client, state in self.s_loc.items():
            delta = state - self.state
            aux_var[client] = {"delta": delta}
        return aux_var

    def process_aux_var(
        self,
        aux_var: Dict[str, Any],
    ) -> None:
        """Update this module based on received shared auxiliary variables.

        Collect updated local state variables sent by clients.
        Update the global state variable based on the latter.
        """
        # Collect updated local states received from Scaffold client modules.
        s_new = {}  # type: Dict[str, Union[Vector, float]]
        for client, c_dict in aux_var.items():
            if not isinstance(c_dict, dict):
                raise TypeError(
                    "ScaffoldServerModule requires auxiliary variables "
                    "to be received as client-wise dictionaries."
                )
            if "state" not in c_dict:
                raise KeyError(
                    "Missing required 'state' key in auxiliary variables "
                    f"received by ScaffoldServerModule from client '{client}'."
                )
            state = c_dict["state"]
            if isinstance(state, float) and state == 0.0:
                # Drop info from clients that have not processed gradients.
                continue
            if isinstance(state, (Vector, float)):
                s_new[client] = state
            else:
                raise TypeError(
                    "Unsupported type for auxiliary variable 'state' "
                    f"received by ScaffoldServerModule from client '{client}'."
                )
        # Update the global and client-wise state variables.
        update = sum(
            state - self.s_loc.get(client, 0.0)
            for client, state in s_new.items()
        )
        self.s_loc.update(s_new)
        update = update / len(self.s_loc)
        self.state = self.state + update

    def get_state(
        self,
    ) -> Dict[str, Any]:
        return {"state": self.state, "s_loc": self.s_loc}

    def set_state(
        self,
        state: Dict[str, Any],
    ) -> None:
        for key in ("state", "s_loc"):
            if key not in state:
                raise KeyError(f"Missing required state variable '{key}'.")
        self.state = state["state"]
        self.s_loc = state["s_loc"]
