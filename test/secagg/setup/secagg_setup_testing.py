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

"""Shared unit tests suite for SecAgg config and setup classes."""

import abc
import asyncio
from typing import Any, Dict, Generic, List, Tuple, Type, TypeVar

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from declearn.secagg.api import (
    Decrypter,
    Encrypter,
    SecaggConfigClient,
    SecaggConfigServer,
    SecaggSetupQuery,
)
from declearn.test_utils import MockNetworkClient, MockNetworkServer


DecrypterT = TypeVar("DecrypterT", bound=Decrypter)
EncrypterT = TypeVar("EncrypterT", bound=Encrypter)
SecaggSetupMsgT = TypeVar("SecaggSetupMsgT", bound=SecaggSetupQuery)


class SecaggSetupTestCase(
    Generic[DecrypterT, EncrypterT, SecaggSetupMsgT],
    metaclass=abc.ABCMeta,
):
    """Base class defining shared SecAgg setup tests."""

    decrypter_cls: Type[Decrypter]
    encrypter_cls: Type[Encrypter]
    client_config_cls: Type[SecaggConfigClient[EncrypterT, SecaggSetupMsgT]]
    server_config_cls: Type[SecaggConfigServer[DecrypterT, SecaggSetupMsgT]]
    setup_msg_cls: Type[SecaggSetupMsgT]

    def test_prepare_secagg_setup_query(
        self,
    ) -> None:
        """Test the 'prepare_secagg_setup_query' method's return type."""
        kwargs = self.get_server_hyper_parameters()
        config = self.server_config_cls(**kwargs)
        query = config.prepare_secagg_setup_query()
        assert isinstance(query, SecaggSetupQuery)
        assert isinstance(query, self.setup_msg_cls)
        assert query.bitsize == kwargs["bitsize"]
        assert query.clipval == kwargs["clipval"]

    async def run_server_routine(
        self,
        n_clients: int,
        **kwargs: Any,
    ) -> DecrypterT:
        """Prepare for and run the server-side setup routine."""
        config = self.server_config_cls(**kwargs)
        async with MockNetworkServer() as netwk:
            await netwk.wait_for_clients(n_clients)
            decrypter = await config.setup_decrypter(netwk)
        return decrypter

    async def run_client_routine(
        self,
        name: str,
        prv_key: Ed25519PrivateKey,
        trusted: List[Ed25519PublicKey],
        **kwargs: Any,
    ) -> EncrypterT:
        """Prepare for and run the client-side setup routine."""
        config = self.client_config_cls.from_params(
            id_keys={"prv_key": prv_key, "trusted": trusted}, **kwargs
        )
        async with MockNetworkClient(name=name) as netwk:
            await netwk.register({})
            msg = await netwk.recv_message()
            encrypter = await config.setup_encrypter(netwk, msg)
        return encrypter

    def get_server_hyper_parameters(
        self,
    ) -> Dict[str, Any]:
        """Return server-side arbitraty config hyper-parameters."""
        return {"bitsize": 16, "clipval": 100.0}

    def get_client_hyper_parameters(
        self,
    ) -> Dict[str, Any]:
        """Return client-side arbitraty config hyper-parameters."""
        return {}

    @pytest.mark.parametrize("n_clients", [2, 5])
    @pytest.mark.asyncio
    async def test_secagg_setup(
        self,
        n_clients: int,
    ) -> None:
        """Test that the SecAgg setup routines work properly."""
        # Generate identity keys.
        id_keys = [Ed25519PrivateKey.generate() for _ in range(n_clients)]
        trusted = [key.public_key() for key in id_keys]
        # Use arbitrary, preferably non-default values.
        server_kwargs = self.get_server_hyper_parameters()
        client_kwargs = self.get_client_hyper_parameters()
        # Setup the server and client routines.
        client_routines = [
            self.run_client_routine(
                f"client_{i}", id_keys[i], trusted, **client_kwargs
            )
            for i in range(n_clients)
        ]
        server_routine = self.run_server_routine(n_clients, **server_kwargs)
        # Run the routines concurrently and gather resulting objects.
        decrypter, *encrypters = await (  # type: ignore[assignment]
            asyncio.gather(server_routine, *client_routines)
        )  # type: Tuple[DecrypterT, List[EncrypterT]]
        # Verify that the resulring objects have proper types and parameters.
        kwargs = {**server_kwargs, **client_kwargs, "n_clients": n_clients}
        self.assert_decrypter_validity(decrypter, **kwargs)
        for encrypter in encrypters:
            self.assert_encrypter_validity(encrypter, decrypter, **kwargs)
        # Verify that encryption/decryption works for arbitrary values.
        encrypted = [encrypter.encrypt_uint(1) for encrypter in encrypters]
        assert all(isinstance(x, int) and (x != 1) for x in encrypted)
        encrypted_sum = self.sum_encrypted(encrypted, **kwargs)
        assert decrypter.decrypt_uint(encrypted_sum) == n_clients

    def assert_decrypter_validity(
        self,
        decrypter: DecrypterT,
        **kwargs: Any,
    ) -> None:
        """Assert that a setup Decrypter matches expectations.

        Subclasses should overload this method to perform
        method-dependent assertions.
        """
        assert isinstance(decrypter, self.decrypter_cls)
        assert decrypter.quantizer.val_range == kwargs["clipval"]

    def assert_encrypter_validity(
        self,
        encrypter: EncrypterT,
        decrypter: DecrypterT,
        **kwargs: Any,
    ) -> None:
        """Assert that a setup Encrypter matches expectations.

        Subclasses should overload this method to perform
        method-dependent assertions.
        """
        # overloadable method; pylint: disable=unused-argument
        assert isinstance(encrypter, self.encrypter_cls)
        assert encrypter.quantizer.int_range == decrypter.quantizer.int_range
        assert encrypter.quantizer.val_range == decrypter.quantizer.val_range

    def sum_encrypted(
        self,
        encrypted: List[int],
        **kwargs: Any,
    ) -> int:
        """Sum encrypted values.

        Subclasses may overload this method when encrypted values are to be
        summed using a dedicated operator rather than mere summation.
        """
        # overloadable method; pylint: disable=unused-argument
        return sum(encrypted)
