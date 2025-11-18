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

"""Handler to hold and load long-lived Ed25519 identity keys."""

import functools
import getpass
from typing import List, Literal, Optional, Sequence, Union

from cryptography.hazmat.primitives import (
    serialization as cryptography_serialization,
)
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.types import (
    PrivateKeyTypes,
    PublicKeyTypes,
)

__all__ = [
    "IdentityKeys",
]


class IdentityKeys:
    """Handler to hold and load long-lived Ed25519 identity keys.

    This class is designed to hold (and possibly import and/or export)
    identity keys that may be used in authenticating and/or encrypting
    messages exchanged across peers.

    It is designed to exclusively use Ed25519 keys, and relies on the
    `cryptography` third-party library to parse, format and interface
    these keys and associated encryption and signature primitives.

    This class was designed with Secure Aggregation (SecAgg) in mind,
    and given the purpose to load and hold long-lived asymmetric keys
    exchanged outside of DecLearn, that may be leveraged into setting
    up ephemeral symmetric secret keys and thereof any kind of secret
    required for SecAgg to be implemented.
    """

    def __init__(
        self,
        prv_key: Union[str, Ed25519PrivateKey],
        trusted: Union[str, Sequence[Union[Ed25519PublicKey, str]]],
        password: Optional[bytes] = None,
    ) -> None:
        """Instantiate the Ed25519 identity keys handler.

        Parameters
        ----------
        prv_key:
            Private Ed25519 identity key for this instance.
            May be a path to a key file with either OpenSSH,
            PEM, DER or raw private bytes format.
        trusted:
            List of trusted public Ed25519 identity keys.
            May be a path to a single custom-format file,
            or a list of keys and/or paths to key files.
        password:
            Optional password required to open the `prv_key` file.
            If a password is required but not provided, a user prompt
            will be used to securely gather it.

        Raises
        ------
        TypeError
            If a provided value has unproper type, or a key file
            stores a key of unproper type.
        ValueError
            If a provided file has unproper format.
        """
        # Load private Ed25519 key.
        if isinstance(prv_key, str):
            prv_key = self.load_ed25519_private_key_from_file(
                path=prv_key, password=password
            )
        if not isinstance(prv_key, Ed25519PrivateKey):
            raise TypeError(
                "'prv_key' must be an Ed25519PrivateKey or path to one."
            )
        self.prv_key = prv_key
        # Load trusted public Ed25519 keys.
        self.trusted = self._load_trusted_keys(trusted)

    @functools.cached_property
    def pub_key(
        self,
    ) -> Ed25519PublicKey:
        """Public identity key matching the held private one."""
        return self.prv_key.public_key()

    @functools.cached_property
    def pub_key_bytes(
        self,
    ) -> bytes:
        """Public bytes associated with this instance's public identity key."""
        return self.pub_key.public_bytes_raw()

    def _load_trusted_keys(
        self,
        trusted: Union[str, Sequence[Union[Ed25519PublicKey, str]]],
    ) -> List[Ed25519PublicKey]:
        """Load trusted public keys from input values."""
        # Case when the path to a single file is received.
        if isinstance(trusted, str):
            return self.load_trusted_keys_from_file(path=trusted)
        # Case when inputs' type is invalid.
        if not isinstance(trusted, (list, tuple)):
            raise TypeError(
                f"'{self.__class__}' requires 'trusted' to be either a str "
                "(path to a file) or a list of strings and/or public keys, "
                f"but it received inputs with type '{type(trusted)}'."
            )
        # Case when inputs are a list of values. Type-check and/or load keys.
        keys: List[Ed25519PublicKey] = []
        for value in trusted:
            if isinstance(value, Ed25519PublicKey):
                keys.append(value)
            elif isinstance(value, str):
                keys.append(self.load_ed25519_public_key_from_file(value))
            else:
                raise TypeError(
                    f"'{self.__class__}' requires 'trusted' list to contain "
                    "Ed25519 public keys and/or paths to such keys, but it "
                    f"received an element with type '{type(value)}'."
                )
        return keys

    @classmethod
    def load_ed25519_private_key_from_file(
        cls,
        path: str,
        password: Optional[bytes] = None,
    ) -> Ed25519PrivateKey:
        """Load a private Ed25519 key from a file.

        Parameters
        ----------
        path:
            Path to an Ed25519 private key file, with either OpenSSH,
            PEM, DER, or raw private bytes format.
        password:
            Optional bytes-formatted password to open the file.
            If a password is required but not provided, a user prompt
            will be used to securely gather it.

        Returns
        -------
        prv_key:
            Private Ed25519 key, wrapped using a `cryptography` type.

        Raises
        ------
        ValueError
            If the file cannot be parsed into a private identity key.
        TypeError
            If the parsed key is not a private Ed25519 one.
        """
        # Read the key file's data.
        with open(path, "rb") as file:
            data = file.read()
        # Attempt to identify its format and thus decode it.
        if data.startswith(b"-----BEGIN OPENSSH PRIVATE KEY"):
            pkey = cls._decode_private_key(data, path, password, "ssh")
        elif (
            path.endswith(".pem")
            or data.startswith(b"-----BEGIN PRIVATE")
            or data.startswith(b"-----BEGIN ENCRYPTED")
        ):
            pkey = cls._decode_private_key(data, path, password, "pem")
        elif len(data) == 32:
            pkey = Ed25519PrivateKey.from_private_bytes(data)
        else:
            pkey = cls._decode_private_key(data, path, password, "der")
        # Verify that the key has proper type.
        if not isinstance(pkey, Ed25519PrivateKey):
            raise TypeError(
                f"Expected to load a Ed25519 private key from file '{path}',"
                f" got a key with type '{type(pkey)}'."
            )
        # Return the loaded key.
        return pkey

    @classmethod
    def _decode_private_key(
        cls,
        data: bytes,
        path: str,
        password: Optional[bytes],
        encoding: Literal["ssh", "pem", "der"],
    ) -> PrivateKeyTypes:
        """Decode a DER-encoded private key from a file's contents.

        Parameters
        ----------
        data:
            Key file contents, as bytes data.
        path:
            Path to the key file from which `data` was read.
            Merely used to contextualize prompts and errors.
        password:
            Optional bytes-encoded password to unlock the key.
            If needed, the user will be prompted for one.
        encoding:
            Type of key serialization encoding.
            One of `{"ssh", "pem", "der"}`.
        """
        # Identify the loading function associated with the format.
        loader = {
            "der": cryptography_serialization.load_der_private_key,
            "pem": cryptography_serialization.load_pem_private_key,
            "ssh": cryptography_serialization.load_ssh_private_key,
        }[encoding]
        # Try to parse the data using the provided (or no) password.
        try:
            return loader(data, password)  # type: ignore[operator]
        except Exception as exc:
            # If a password is missing, prompt for one and retry.
            if password is None and exc.args and ("encrypted" in exc.args[0]):
                password = cls._prompt_key_password(path)
                return cls._decode_private_key(data, path, password, encoding)
            # Wrap any other exception as a ValueError.
            raise ValueError(
                f"Failed to parse a private key from file '{path}' with "
                f"identified format '{encoding.upper()}'."
            ) from exc

    @staticmethod
    def _prompt_key_password(
        path: str,
    ) -> bytes:
        """Prompt the user for the password to a private key file."""
        try:
            return getpass.getpass(
                f"Enter password to unlock private key file '{path}':\n"
            ).encode("utf-8")
        except EOFError as exc:
            raise RuntimeError(
                f"Unable to prompt for a password to unlock key file '{path}'."
            ) from exc

    @classmethod
    def load_ed25519_public_key_from_file(
        cls,
        path: str,
    ) -> Ed25519PublicKey:
        """Load a public Ed25519 key from a file.

        Parameters
        ----------
        path:
            Path to an Ed25519 public key file, with either OpenSSH,
            PEM, DER or raw public bytes format.

        Returns
        -------
        pub_key:
            Public Ed25519 key, wrapped using a `cryptography` type.

        Raises
        ------
        ValueError
            If the file cannot be parsed into a public identity key.
        TypeError
            If the parsed key is not a public Ed25519 one.
        """
        # Read the key file's data.
        with open(path, "rb") as file:
            data = file.read()
        # Attempt to identify its format and thus decode it.
        if data.startswith(b"ssh-"):
            pkey = cls._decode_public_key(data, path, encoding="ssh")
        elif path.endswith(".pem") or data.startswith(b"-----BEGIN PUBLIC"):
            pkey = cls._decode_public_key(data, path, encoding="pem")
        elif len(data) == 32:
            pkey = Ed25519PublicKey.from_public_bytes(data)
        else:
            pkey = cls._decode_public_key(data, path, encoding="der")
        # Verify that the key has proper type.
        if not isinstance(pkey, Ed25519PublicKey):
            raise TypeError(
                f"Expected to load a Ed25519 public key from file '{path}',"
                f" got a key with type '{type(pkey)}'."
            )
        # Return the loaded key.
        return pkey

    @staticmethod
    def _decode_public_key(
        data: bytes,
        path: str,
        encoding: Literal["ssh", "pem", "der"],
    ) -> PublicKeyTypes:
        """Decode a DER-encoded private key from a file's contents.

        Parameters
        ----------
        data:
            Key file contents, as bytes data.
        path:
            Path to the key file from which `data` was read.
            Merely used to contextualize errors.
        encoding:
            Type of key serialization encoding.
            One of `{"ssh", "pem", "der"}`.
        """
        # Identify the loading function associated with the format.
        loader = {
            "der": cryptography_serialization.load_der_public_key,
            "pem": cryptography_serialization.load_pem_public_key,
            "ssh": cryptography_serialization.load_ssh_public_key,
        }[encoding]
        # Try to parse the data using the provided (or no) password.
        try:
            return loader(data)
        # Wrap any exception as a ValueError.
        except Exception as exc:
            raise ValueError(
                f"Failed to parse a public key from file '{path}' with "
                f"identified format '{encoding.upper()}'."
            ) from exc

    @staticmethod
    def load_trusted_keys_from_file(
        path: str,
    ) -> List[Ed25519PublicKey]:
        """Read multiple Ed25519 public keys from a single file.

        Expect the `export_trusted_ed25519_keys_to_file` custom format,
        i.e. raw bytes from keys stacked with a simple line delimiter.

        Parameters
        ----------
        path:
            Path to the file where the trusted public keys are
            stored.

        Returns
        -------
        trusted:
            List of trusted Ed25519 public keys, wrapped using
            the `Ed25519PublicKey` type from `cryptography`.

        Raises
        ------
        ValueError
            If the file is not found or cannot be parsed.
        """
        keys: List[Ed25519PublicKey] = []
        try:
            with open(path, "rb") as file:
                while dat := file.read(33):
                    key = Ed25519PublicKey.from_public_bytes(dat[:32])
                    keys.append(key)
        except BaseException as exc:
            raise ValueError(
                "Failed to parse trusted Ed25519 public keys from file "
                f"'{path}'."
            ) from exc
        return keys

    def export_trusted_keys_to_file(
        self,
        path: str,
    ) -> None:
        """Export wrapped trusted Ed25519 public keys to a single file.

        Use a custom (and simple) file format, that can be parsed using
        the `load_trusted_keys_from_file` method from this instance into
        the initial list of public key `cryptography` objects.

        Parameters
        ----------
        path:
            Path to the file where the held trusted public keys are
            to be exported.
        """
        data = b"\n".join(key.public_bytes_raw() for key in self.trusted)
        with open(path, "wb") as file:
            file.write(data)
