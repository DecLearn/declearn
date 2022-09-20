# coding: utf-8

"""Shared fixtures for declearn.communication module testing."""

import os
import shlex
import subprocess
import tempfile
from typing import Dict, Iterator, Optional, Tuple

import pytest


@pytest.fixture(name="ssl_cert", scope="module")
def ssl_cert_fixture() -> Iterator[Dict[str, str]]:
    """Fixture providing path to temporary self-signed TLS/SSL certificates.

    Return a dict containing paths to PEM files for a client (CA)
    certificate, a server certificate and a server private key,
    without any password protection.

    These files are located in a temporary directory that is deleted
    once this fixture goes out of scope. This fixture should be made
    to have a large scope to avoid unrequired repeated OpenSSL calls
    between tests.
    """
    with tempfile.TemporaryDirectory() as folder:
        ca_cert, sv_cert, sv_priv = generate_ssl_certificates(folder)
        yield {
            "client_cert": ca_cert,
            "server_cert": sv_cert,
            "server_pkey": sv_priv,
        }


def generate_ssl_certificates(
        folder: str = ".",
        c_name: str = "localhost",
        password: Optional[str] = None
    ) -> Tuple[str, str, str]:
    """Generate self-signed certificates for localhost testing purposes.

    Parameters
    ----------
    folder: str
        Path to the (ideally, temporary) folder where to create files.
    c_name: str
        CommonName value for the server certificate.
    password: str or None, default=None
        Optional password used to encrypt generated private keys.

    Returns
    -------
    ca_cert: str
        Path to the client-required CA certificate PEM file.
    sv_cert: str
        Path to the server's certificate PEM file.
    sv_priv: str
        Path to the server's private key PEM file (password: "supertoto").
    """
    # Generate self-signed CA certificate and private key.
    ca_priv = os.path.join(folder, "ca-key.pem")
    ca_cert = os.path.join(folder, "ca-cert.pem")
    cmd = (
        "openssl req -x509 -newkey rsa:4096 "
        + f"-keyout {ca_priv} -out {ca_cert} "
        + (f"-passout pass:{password} " if password else "-nodes ")
        + "-sha256 -days 365 "
        + '-subj "/C=FR/L=Lille/O=Inria/OU=Magnet/CN=inria.fr"'
    )
    subprocess.run(shlex.split(cmd), check=True, capture_output=True)
    # Generate server private key and CSR (certificate signing request).
    sv_priv = os.path.join(folder, "server-key.pem")
    sv_csrq = os.path.join(folder, "server-req.pem")
    cmd = (
        "openssl req -newkey rsa:4096 "
        + f"-keyout {sv_priv} -out {sv_csrq} "
        + (f"-passout pass:{password} " if password else "-nodes ")
        + f'-subj "/C=FR/L=Lille/O=Inria/OU=Magnet/CN={c_name}"'
    )
    subprocess.run(shlex.split(cmd), check=True, capture_output=True)
    # Generate self-signed server certificate.
    sv_cert = os.path.join(folder, "server-cert.pem")
    cmd = (
        f"openssl x509 -req -in {sv_csrq} -CA {ca_cert} "
        + f"-CAkey {ca_priv} -CAcreateserial -out {sv_cert} "
        + (f"-passin pass:{password} " if password else " ")
        + "-days 30"
    )
    subprocess.run(shlex.split(cmd), check=True, capture_output=True)
    # Return paths that are used in tests.
    return ca_cert, sv_cert, sv_priv
