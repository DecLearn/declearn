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

import abc
import dataclasses
from typing import ClassVar, Generic, Optional, Set, TypeVar

from declearn.communication.api import NetworkClient, NetworkServer
from declearn.messaging import Message, SerializedMessage
from declearn.secagg.api._decrypt import Decrypter
from declearn.secagg.api._encrypt import Encrypter
from declearn.secagg.utils import IdentityKeys
from declearn.utils import TomlConfig, create_types_registry, register_type

__all__ = [
    "SecaggConfigClient",
    "SecaggConfigServer",
    "SecaggSetupQuery",
]


@dataclasses.dataclass
class SecaggSetupQuery(Message, register=False, metaclass=abc.ABCMeta):
    """ABC message for all SecAgg setup init requests.

    This message should be subclassed into SecAgg-protocol-specific
    setup requests. By default, it contains server-set quantization
    hyper-parameters.
    """

    bitsize: int
    clipval: float


DecrypterT = TypeVar("DecrypterT", bound=Decrypter)
EncrypterT = TypeVar("EncrypterT", bound=Encrypter)
SecaggSetupMsgT = TypeVar("SecaggSetupMsgT", bound=SecaggSetupQuery)


@create_types_registry(name="SecaggConfigClient")
@dataclasses.dataclass
class SecaggConfigClient(
    TomlConfig,
    Generic[EncrypterT, SecaggSetupMsgT],
    metaclass=abc.ABCMeta,
):
    """ABC for client-side SecAgg configuration and setup.

    This class defines an API for configuring the use of SecAgg on the
    client side, and is two-fold. On the one hand, it is a dataclass
    that can be parsed from TOML or kwargs, enabling to specify that
    SecAgg should be used and on what grounds. On the other hand, it
    exposes the `setup_encrypter` async method that should be called
    upon receiving a query from the server to participate in a SecAgg
    setup protocol and return an `Encrypter` matching the config.

    The `secagg_type` class attribute must be defined by subclasses,
    and paired server/client classes are expected to share the same
    name.

    Fields
    ------
    id_keys:
        `IdentityKeys` handler holding long-lived identity keys.
        This may be specified as a dict (notably in TOML files),
        with the following fields:
            - `prv_key`: path to a private ed25519 key.
            - `trusted`: path or list of paths to trusted peers'
              public ed25519 keys.
            - (opt.) `password`: optional password to decrypt the
              private key file; if required, a user prompt may be
              used rather than passing the password in clear.
    """

    id_keys: IdentityKeys

    secagg_type: ClassVar[str]

    def __init_subclass__(
        cls,
        register: bool = True,
    ) -> None:
        if register:
            register_type(cls, cls.secagg_type, group="SecaggConfigClient")

    @abc.abstractmethod
    async def setup_encrypter(
        self,
        netwk: NetworkClient,
        query: SerializedMessage[SecaggSetupMsgT],
    ) -> EncrypterT:
        """Setup a SecAgg Encrypter based on a server-emitted query.

        Parameters
        ----------
        netwk:
            `NetworkClient` instance to be used to communicate with
            the server. Expected to be started and registered.
        query:
            `SerializedMessage` wrapping a query to setup SecAgg,
            freshly received from the server.

        Returns
        -------
        encrypter:
            `Encrypter` instance, the exact type of which depends on
            the SecAgg method being set up.

        Raises
        ------
        KeyError
            If a peer's public identity key is not part of trusted ones.
            If any cryptographic key is misused as part of the process.
        RuntimeError
            If the protocol fails due to the server or peers not following
            expected steps or raising errors themselves.
        ValueError
            If a signature verification fails, that may indicate tempering.
        """


@create_types_registry(name="SecaggConfigServer")
@dataclasses.dataclass
class SecaggConfigServer(
    TomlConfig,
    Generic[DecrypterT, SecaggSetupMsgT],
    metaclass=abc.ABCMeta,
):
    """ABC for server-side SecAgg configuration and setup.

    This class defines an API for configuring the use of SecAgg on the
    server side, and is two-fold. On the one hand, it is a dataclass
    that can be parsed from TOML or kwargs, enabling to specify that
    SecAgg should be used and on what grounds. On the other hand, it
    exposes the `setup_decrypter` async method that should be called
    to trigger a protocol involving (a subset of) clients resulting
    in setting up client-wise `Encrypter`s and returning a matching
    `Decrypter`, that abide by the configured SecAgg method.

    The `secagg_type` class attribute must be defined by subclasses,
    and paired server/client classes are expected to share the same
    name.

    Fields
    ------
    bitsize:
        Quantization hyper-parameter, defining the range of output
        quantized integers.
    clipval:
        Quantization hyper-parameter, defining a maximum absolute
        value for floating point numbers being (un)quantized.
    """

    bitsize: int
    clipval: float

    secagg_type: ClassVar[str]

    def __init_subclass__(
        cls,
        register: bool = True,
    ) -> None:
        if register:
            register_type(cls, cls.secagg_type, group="SecaggConfigServer")

    async def setup_decrypter(
        self,
        netwk: NetworkServer,
        clients: Optional[Set[str]] = None,
    ) -> DecrypterT:
        """Orchestrate a SecAgg setup protocol and return a Decrypter.

        Send a query to (selected) clients so as to set up SecAgg anew.
        Have a method-dependent setup be performed together with these
        clients, resulting in the setup of properly-parametrized SecAgg
        controllers. Return the server-held `Decrypter`.

        Parameters
        ----------
        netwk:
            `NetworkServer` communication endpoint, that is expected
            to be running and already have clients registered to it.
        clients:
            Optional subset of clients to which to restrict the setup.

        Returns
        -------
        decrypter:
            SecAgg decryption controller, parametrized to match clients'
            encrypter instances, based on this config object's type and
            attributes.

        Raises
        ------
        RuntimeError
            If the setup protocol fails for any reason.
            This may be re-raised from a more verbose/specific exception.
        """
        # Setup and broadcast a SecAgg setup query to selected clients.
        query = self.prepare_secagg_setup_query()
        await netwk.broadcast_message(query, clients)
        # Perform protocol-specific actions.
        try:
            return await self.finalize_secagg_setup(netwk, clients)
        except Exception as exc:
            raise RuntimeError("SecAgg setup protocol failed.") from exc

    @abc.abstractmethod
    def prepare_secagg_setup_query(
        self,
    ) -> SecaggSetupMsgT:
        """Return a request to setup SecAgg, broadcastable to clients.

        Returns
        -------
        message:
            `SecaggSetupQuery` subclass instance to be sent to clients
            in order to trigger a SecAgg setup protocol.
        """

    @abc.abstractmethod
    async def finalize_secagg_setup(
        self,
        netwk: NetworkServer,
        clients: Optional[Set[str]] = None,
    ) -> DecrypterT:
        """Finalize a SecAgg setup routine, having broadcasted a query.

        This method is the main backend of `setup_decrypter`, that
        should be defined by concrete subclasses to implement setup
        behavior once the initial `SecaggSetupQuery` has been sent
        to `clients`.
        """
