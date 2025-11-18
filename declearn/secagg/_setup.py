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

"""API-defining ABCs for SecAgg setup config, routines and messages."""

from typing import Any, Dict, Tuple, Type, Union

from declearn.secagg.api import SecaggConfigClient, SecaggConfigServer
from declearn.secagg.utils import IdentityKeys
from declearn.utils import access_registered, access_types_mapping

__all__ = [
    "list_available_secagg_types",
    "parse_secagg_config_client",
    "parse_secagg_config_server",
]


def parse_secagg_config_client(
    secagg_type: str,
    id_keys: Union[IdentityKeys, Dict[str, Any]],
    **kwargs: Any,
) -> SecaggConfigClient:
    """Parse input arguments into a `SecaggConfigClient` instance.

    Parameters
    ----------
    secagg_type:
        Name of the SecAgg protocol (matching the target class's
        `secagg_type` attribute, under which it is type-registered).
    id_keys:
        `IdentityKeys` instance or dict of kwargs to instantiate one,
        holding this client's private identity key and public identity
        keys of trusted clients.
    **kwargs:
        Any additional method-specific keyword argument may be passed.

    Returns
    -------
    secagg_config:
        `SecaggConfigClient` instance parameterized based on inputs.
    """
    cls = access_registered(secagg_type, group="SecaggConfigClient")
    assert issubclass(cls, SecaggConfigClient)
    return cls.from_params(id_keys=id_keys, **kwargs)


def parse_secagg_config_server(
    secagg_type: str,
    bitsize: int = 64,
    clipval: float = 1e5,
    **kwargs: Any,
) -> SecaggConfigServer:
    """Parse input arguments into a `SecaggConfigServer` instance.

    Parameters
    ----------
    secagg_type:
        Name of the SecAgg protocol (matching the target class's
        `secagg_type` attribute, under which it is type-registered).
    bitsize:
        Quantization hyper-parameter, defining the range of output
        quantized integers. Set to use `uint64` values by default.
    clipval:
        Quantization hyper-parameter, defining a maximum absolute
        value for floating point numbers being (un)quantized.
        This is arbitrarily set to `10^5` by default but should be
        adjusted based on the application (namely, on the expected
        maximum value of gradients or metrics, at a client's level
        and after aggregation over participating clients).
    **kwargs:
        Any additional method-specific keyword argument may be passed.

    Returns
    -------
    secagg_config:
        `SecaggConfigServer` instance parameterized based on inputs.
    """
    cls = access_registered(secagg_type, group="SecaggConfigServer")
    assert issubclass(cls, SecaggConfigServer)
    return cls.from_params(bitsize=bitsize, clipval=clipval, **kwargs)


def list_available_secagg_types() -> Dict[
    str, Tuple[Type[SecaggConfigClient], Type[SecaggConfigServer]]
]:
    """List available SecAgg types and access associated config types.

    Note: partially-defined SecAgg types (e.g. with a registered type for
    clients but not for the server, or reciprocally) will not be included
    in the outputs.

    Returns
    -------
    secagg_types:
        `{name: (client_type, server_type)}` dict mapping `secagg_type`
        names (that may be used in `parse_secagg_config_(client|server)`)
        to a tuple of corresponding config and setup controller types.
    """
    client = access_types_mapping("SecaggConfigClient")
    server = access_types_mapping("SecaggConfigServer")
    return {
        secagg_type: (client[secagg_type], server[secagg_type])
        for secagg_type in set(client).intersection(server)
    }
