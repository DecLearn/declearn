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

"""Unit tests for 'declearn.secagg.utils.IdentityKeys'."""

import os
from typing import Literal
from unittest import mock

import pytest
from cryptography.hazmat.primitives import (
    serialization as cryptography_serialization,
)
from cryptography.hazmat.primitives.asymmetric.ed448 import Ed448PrivateKey
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.types import (
    PrivateKeyTypes,
    PublicKeyTypes,
)

from declearn.secagg.utils import IdentityKeys

PRIVATE_ENCODING = {
    "ssh": cryptography_serialization.Encoding.PEM,
    "pem": cryptography_serialization.Encoding.PEM,
    "der": cryptography_serialization.Encoding.DER,
    "raw": cryptography_serialization.Encoding.Raw,
}

PRIVATE_FORMAT = {
    "ssh": cryptography_serialization.PrivateFormat.OpenSSH,
    "pem": cryptography_serialization.PrivateFormat.PKCS8,
    "der": cryptography_serialization.PrivateFormat.PKCS8,
    "raw": cryptography_serialization.PrivateFormat.Raw,
}

PUBLIC_ENCODING = {
    "ssh": cryptography_serialization.Encoding.OpenSSH,
    "pem": cryptography_serialization.Encoding.PEM,
    "der": cryptography_serialization.Encoding.DER,
    "raw": cryptography_serialization.Encoding.Raw,
}

PUBLIC_FORMAT = {
    "ssh": cryptography_serialization.PublicFormat.OpenSSH,
    "pem": cryptography_serialization.PublicFormat.SubjectPublicKeyInfo,
    "der": cryptography_serialization.PublicFormat.SubjectPublicKeyInfo,
    "raw": cryptography_serialization.PublicFormat.Raw,
}

PASSWORD = b"password"


@pytest.fixture(name="filepath")
def filepath_fixture(tmp_path: str) -> str:
    """Fixture providing with a temporaty filepath."""
    return os.path.join(tmp_path, "tempfile")


class TestLoadEd25519PrivateKey:
    """Unit tests for 'IdentityKeys.load_ed25519_private_key_from_file'."""

    @staticmethod
    def dump_private_key(
        prv_key: PrivateKeyTypes,
        encoding: Literal["ssh", "pem", "der", "raw"],
        encrypted: bool,
        path: str,
    ) -> None:
        """Dump a private key to a file."""
        encryption = (
            cryptography_serialization.BestAvailableEncryption(PASSWORD)
            if encrypted
            else cryptography_serialization.NoEncryption()
        )
        data = prv_key.private_bytes(
            encoding=PRIVATE_ENCODING[encoding],
            format=PRIVATE_FORMAT[encoding],
            encryption_algorithm=encryption,
        )
        with open(path, "wb") as file:
            file.write(data)

    @pytest.mark.parametrize("encoding", ["ssh", "pem", "der", "raw"])
    def test_load_ed25519_private_key_from_unencrypted_file(
        self,
        encoding: Literal["ssh", "pem", "der", "raw"],
        filepath: str,
    ) -> None:
        """Test that loading a key from an unencrypted file works."""
        key = Ed25519PrivateKey.generate()
        self.dump_private_key(key, encoding, encrypted=False, path=filepath)
        bis = IdentityKeys.load_ed25519_private_key_from_file(
            filepath, password=None
        )
        assert isinstance(bis, Ed25519PrivateKey)
        assert bis.private_bytes_raw() == key.private_bytes_raw()

    @pytest.mark.parametrize("encoding", ["pem", "der"])
    def test_load_ed25519_private_key_from_encrypted_file_with_password(
        self,
        encoding: Literal["ssh", "pem", "der", "raw"],
        filepath: str,
    ) -> None:
        """Test reloading a key from an encrypted file with password."""
        key = Ed25519PrivateKey.generate()
        self.dump_private_key(key, encoding, encrypted=True, path=filepath)
        bis = IdentityKeys.load_ed25519_private_key_from_file(
            filepath, password=PASSWORD
        )
        assert isinstance(bis, Ed25519PrivateKey)
        assert bis.private_bytes_raw() == key.private_bytes_raw()

    @pytest.mark.parametrize("encoding", ["pem", "der"])
    def test_load_ed25519_private_key_from_encrypted_file_without_password(
        self,
        encoding: Literal["ssh", "pem", "der", "raw"],
        filepath: str,
    ) -> None:
        """Test reloading a key from an encrypted file with password prompt."""
        key = Ed25519PrivateKey.generate()
        self.dump_private_key(key, encoding, encrypted=True, path=filepath)
        with mock.patch(
            "getpass.getpass", return_value=PASSWORD.decode()
        ) as patch_getpass:
            bis = IdentityKeys.load_ed25519_private_key_from_file(
                filepath, password=None
            )
        patch_getpass.assert_called_once()
        assert isinstance(bis, Ed25519PrivateKey)
        assert bis.private_bytes_raw() == key.private_bytes_raw()

    @pytest.mark.parametrize("encoding", ["pem", "der"])
    def test_load_ed25519_private_key_from_encrypted_file_wrong_password(
        self,
        encoding: Literal["ssh", "pem", "der", "raw"],
        filepath: str,
    ) -> None:
        """Test reloading a key from an encrypted file with wrong password."""
        key = Ed25519PrivateKey.generate()
        self.dump_private_key(key, encoding, encrypted=True, path=filepath)
        with pytest.raises(ValueError):
            IdentityKeys.load_ed25519_private_key_from_file(
                filepath, password=PASSWORD + b"-wrong"
            )

    @pytest.mark.parametrize("encoding", ["pem", "der"])
    def test_load_ed25519_private_key_from_encrypted_file_wrong_prompt(
        self,
        encoding: Literal["ssh", "pem", "der", "raw"],
        filepath: str,
    ) -> None:
        """Test reloading a key from an encrypted file with wrong pwd input."""
        key = Ed25519PrivateKey.generate()
        self.dump_private_key(key, encoding, encrypted=True, path=filepath)
        with mock.patch(
            "getpass.getpass", return_value=PASSWORD.decode() + "-wrong"
        ) as patch_getpass:
            with pytest.raises(ValueError):
                IdentityKeys.load_ed25519_private_key_from_file(
                    filepath, password=None
                )
        patch_getpass.assert_called_once()

    @pytest.mark.parametrize("encoding", ["pem"])
    def test_load_ed25519_private_key_wrong_type(
        self,
        encoding: Literal["ssh", "pem", "der", "raw"],
        filepath: str,
    ) -> None:
        """Test that loading a non-Ed25519 private key raises a TypeError."""
        key = Ed448PrivateKey.generate()
        self.dump_private_key(key, encoding, encrypted=False, path=filepath)
        with pytest.raises(TypeError):
            IdentityKeys.load_ed25519_private_key_from_file(
                filepath, password=None
            )

    @pytest.mark.parametrize("encoding", ["pem", "der"])
    def test_load_ed25519_private_key_missing_password(
        self,
        encoding: Literal["ssh", "pem", "der", "raw"],
        filepath: str,
    ) -> None:
        """Test that failing to prompt for a password raises a RuntimeError."""
        key = Ed25519PrivateKey.generate()
        self.dump_private_key(key, encoding, encrypted=True, path=filepath)
        with mock.patch("getpass.getpass", side_effect=EOFError) as patch:
            with pytest.raises(RuntimeError):
                IdentityKeys.load_ed25519_private_key_from_file(
                    filepath, password=None
                )
            patch.assert_called_once()


class TestLoadEd25519PublicKey:
    """Unit tests for 'IdentityKeys.load_ed25519_public_key_from_file'."""

    @staticmethod
    def dump_public_key(
        pub_key: PublicKeyTypes,
        encoding: Literal["ssh", "pem", "der", "raw"],
        path: str,
    ) -> None:
        """Dump a private key to a file."""
        data = pub_key.public_bytes(
            encoding=PUBLIC_ENCODING[encoding],
            format=PUBLIC_FORMAT[encoding],
        )
        with open(path, "wb") as file:
            file.write(data)

    @pytest.mark.parametrize("encoding", ["ssh", "pem", "der", "raw"])
    def test_load_ed25519_public_key_from_file(
        self,
        encoding: Literal["ssh", "pem", "der", "raw"],
        filepath: str,
    ) -> None:
        """Test that loading a public key from a file works."""
        key = Ed25519PrivateKey.generate().public_key()
        self.dump_public_key(key, encoding, path=filepath)
        bis = IdentityKeys.load_ed25519_public_key_from_file(filepath)
        assert isinstance(bis, Ed25519PublicKey)
        assert bis.public_bytes_raw() == key.public_bytes_raw()

    @pytest.mark.parametrize("encoding", ["pem"])
    def test_load_ed25519_public_key_wrong_type(
        self,
        encoding: Literal["ssh", "pem", "der", "raw"],
        filepath: str,
    ) -> None:
        """Test that loading a non-Ed25519 public key raises a TypeError."""
        key = Ed448PrivateKey.generate().public_key()
        self.dump_public_key(key, encoding, path=filepath)
        with pytest.raises(TypeError):
            IdentityKeys.load_ed25519_public_key_from_file(filepath)

    def test_load_ed25519_public_key_invalid_data(
        self,
        filepath: str,
    ) -> None:
        """Test that loading a malformed public key raises a ValueError."""
        # Export the file twice to the same file, resulting in an invalid dump.
        key = Ed448PrivateKey.generate().public_key()
        self.dump_public_key(key, encoding="raw", path=filepath)
        with open(filepath, "ab") as file:
            file.write(key.public_bytes_raw())
        # Verify that this raises a ValueError due to parsing failing.
        with pytest.raises(ValueError):
            IdentityKeys.load_ed25519_public_key_from_file(filepath)


class TestIdentityKeys:
    """Unit tests for 'declearn.secagg.utils.IdentityKeys'."""

    def test_instantiate_from_key_objects(
        self,
    ) -> None:
        """Test instantiating from Ed25519 key objects."""
        prv_key = Ed25519PrivateKey.generate()
        trusted = [Ed25519PrivateKey.generate().public_key() for _ in range(2)]
        id_keys = IdentityKeys(prv_key=prv_key, trusted=trusted)
        assert id_keys.prv_key == prv_key
        assert id_keys.trusted == trusted

    def test_instantiate_from_key_paths(
        self,
        tmp_path: str,
    ) -> None:
        """Test instantiating from Ed25519 file dumps."""
        # Generate random private and public keys.
        prv_key = Ed25519PrivateKey.generate()
        trusted = [Ed25519PrivateKey.generate().public_key() for _ in range(2)]
        # Dump them to files.
        prv_path = os.path.join(tmp_path, "prv")
        pub_paths = [os.path.join(tmp_path, f"pub_{i}") for i in range(2)]
        TestLoadEd25519PrivateKey.dump_private_key(
            prv_key, "raw", encrypted=False, path=os.path.join(tmp_path, "prv")
        )
        for key, path in zip(trusted, pub_paths, strict=False):
            TestLoadEd25519PublicKey.dump_public_key(key, "raw", path)
        # Instantiate a wrapper from paths.
        id_keys = IdentityKeys(prv_key=prv_path, trusted=pub_paths)
        assert (
            id_keys.prv_key.private_bytes_raw() == prv_key.private_bytes_raw()
        )
        assert all(
            key_a.public_bytes_raw() == key_b.public_bytes_raw()
            for key_a, key_b in zip(id_keys.trusted, trusted, strict=False)
        )

    def test_export_and_load_trusted_keys_to_and_from_file(
        self,
        tmp_path: str,
    ) -> None:
        """Test that exporting trusted keys to a single file works."""
        # Set up a wrapper around some random keys.
        prv_key = Ed25519PrivateKey.generate()
        trusted = [Ed25519PrivateKey.generate().public_key() for _ in range(2)]
        id_keys = IdentityKeys(prv_key=prv_key, trusted=trusted)
        # Export wrapped keys.
        path = os.path.join(tmp_path, "trusted.keys")
        id_keys.export_trusted_keys_to_file(path)
        assert os.path.isfile(path)
        # Reload wrapped keys.
        loaded = IdentityKeys.load_trusted_keys_from_file(path)
        assert isinstance(loaded, list) and len(loaded) == len(trusted)
        assert all(
            key_a.public_bytes_raw() == key_b.public_bytes_raw()
            for key_a, key_b in zip(loaded, trusted, strict=False)
        )

    def test_load_trusted_keys_from_file_invalid_data(
        self,
        tmp_path: str,
    ) -> None:
        """Test that attempting to load malformed data raises a ValueError."""
        path = os.path.join(tmp_path, "trusted.fake")
        with open(path, "wb") as file:
            file.write(b"mock data")
        with pytest.raises(ValueError):
            IdentityKeys.load_trusted_keys_from_file(path)

    def test_instantiate_from_trusted_keys_file(
        self,
        tmp_path: str,
    ) -> None:
        """Test instantiating using a custom file dump of trusted keys."""
        # Set up a wrapper around some random keys.
        prv_key = Ed25519PrivateKey.generate()
        trusted = [Ed25519PrivateKey.generate().public_key() for _ in range(5)]
        id_keys = IdentityKeys(prv_key=prv_key, trusted=trusted)
        # Export wrapped keys.
        path = os.path.join(tmp_path, "trusted.keys")
        id_keys.export_trusted_keys_to_file(path)
        # Instantiate again from the keys' file.
        id_keys = IdentityKeys(prv_key=prv_key, trusted=path)
        assert all(
            key_a.public_bytes_raw() == key_b.public_bytes_raw()
            for key_a, key_b in zip(id_keys.trusted, trusted, strict=False)
        )

    def test_instantiate_with_invalid_private_key_type(
        self,
    ) -> None:
        """Test that providing with a wrong private key raises a TypeError."""
        prv_key = Ed448PrivateKey.generate()
        trusted = [Ed25519PrivateKey.generate().public_key() for _ in range(2)]
        with pytest.raises(TypeError):
            IdentityKeys(prv_key=prv_key, trusted=trusted)  # type: ignore

    def test_instantiate_with_invalid_public_key_type(
        self,
    ) -> None:
        """Test that providing with a wrong public key raises a TypeError."""
        prv_key = Ed25519PrivateKey.generate()
        trusted = [
            Ed25519PrivateKey.generate().public_key(),
            Ed448PrivateKey.generate().public_key(),
        ]
        with pytest.raises(TypeError):
            IdentityKeys(prv_key=prv_key, trusted=trusted)  # type: ignore

    def test_instantiate_with_invalid_trusted_keys_type(
        self,
    ) -> None:
        """Test that providing a wonrg-type 'trusted' raises a TypeError."""
        prv_key = Ed25519PrivateKey.generate()
        trusted = Ed25519PrivateKey.generate().public_key()
        with pytest.raises(TypeError):
            IdentityKeys(prv_key=prv_key, trusted=trusted)  # type: ignore
