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

"""Shared fixtures for declearn.communication module testing."""

import os
import shlex
import subprocess
from typing import Collection, Optional, Tuple


__all__ = [
    "generate_ssl_certificates",
]


def generate_ssl_certificates(
    folder: str = ".",
    c_name: str = "localhost",
    password: Optional[str] = None,
    alt_ips: Optional[Collection[str]] = None,
    alt_dns: Optional[Collection[str]] = None,
) -> Tuple[str, str, str]:
    """Generate a self-signed CA and a CA-signed SSL certificate.

    This function is intended to be used for testing and/or in
    demonstration contexts, whereas real-life applications are
    expected to use certificates signed by a trusted CA.

    This functions orchestrates calls to the system's `openssl`
    command in order to generate and self-sign SSL certificate
    and private-key files that may be used to encrypt network
    communications, notably for declearn.

    More precisely, it generates:
    - a self-signed root certificate authority (CA)
    - a server certificate signed by the former CA

    Parameters
    ----------
    folder: str
        Path to the folder where to create the intermediate
        and final certificate and key PEM files.
    c_name: str
        Main domain name or IP for which the certificate is created.
    password: str or None, default=None
        Optional password used to encrypt generated private keys.
    alt_ips: collection[str] or None, default=None
        Optional list of additional IP addresses to certify.
        This is only implemented for OpenSSL >= 3.0.
    alt_dns: collection[str] or None, default=None
        Optional list of additional domain names to certify.
        This is only implemented for OpenSSL >= 3.0.

    Returns
    -------
    ca_cert: str
        Path to the client-required CA certificate PEM file.
    sv_cert: str
        Path to the server's certificate PEM file.
    sv_pkey: str
        Path to the server's private key PEM file.
    """
    try:
        proc = subprocess.run(
            ["openssl", "version"], check=True, capture_output=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("Failed to parse openssl version.") from exc
    old = proc.stdout.decode().startswith("OpenSSL 1")
    if (alt_ips or alt_dns) and old:
        raise RuntimeError(
            "Cannot add subject alternative names with OpenSSL version <3.0."
        )
    # Generate a self-signed root CA.
    ca_cert, ca_pkey = gen_ssl_ca(folder, password)
    # Generate a server CSR and a private key.
    sv_csrq, sv_pkey = gen_ssl_csr(folder, c_name, alt_ips, alt_dns, password)
    # Sign the CSR into a server certificate using the root CA.
    sv_cert = gen_ssl_cert(folder, sv_csrq, ca_cert, ca_pkey, password, old)
    # Return paths that are used by declearn network-communication endpoints.
    return ca_cert, sv_cert, sv_pkey


def gen_ssl_ca(
    folder: str,
    password: Optional[str] = None,
) -> Tuple[str, str]:
    """Generate a self-signed CA certificate and its private key."""
    # Set up the command to generate the self-signed CA and its key.
    ca_pkey = os.path.join(folder, "ca-pkey.pem")
    ca_cert = os.path.join(folder, "ca-cert.pem")
    cmd = (
        "openssl req -x509 -newkey rsa:4096 -sha256 -days 365 "
        + f"-keyout {ca_pkey} -out {ca_cert} "
        + (f"-passout pass:{password} " if password else "-nodes ")
        + '-subj "/C=FR/L=Lille/O=Inria/OU=Magnet/CN=SelfSignedCA"'
    )
    # Run the command and return the paths to the created files.
    subprocess.run(shlex.split(cmd), check=True, capture_output=True)
    return ca_cert, ca_pkey


def gen_ssl_csr(
    folder: str,
    c_name: str,
    alt_ips: Optional[Collection[str]] = None,
    alt_dns: Optional[Collection[str]] = None,
    password: Optional[str] = None,
) -> Tuple[str, str]:
    """Generate a CSR (certificate signing request) and its private key."""
    sv_pkey = os.path.join(folder, "server-pkey.pem")
    sv_csrq = os.path.join(folder, "server-csrq.pem")
    cmd = (
        "openssl req -newkey rsa:4096 "
        + f"-keyout {sv_pkey} -out {sv_csrq} "
        + f"-subj /C=FR/L=Lille/O=Inria/OU=Magnet/CN={c_name}"
        + (f" -passout pass:{password}" if password else " -nodes")
    )
    alt_names = [f"IP.{i}:{x}" for i, x in enumerate(alt_ips or tuple(), 1)]
    alt_names += [f"DNS.{i}:{x}" for i, x in enumerate(alt_dns or tuple(), 1)]
    if alt_names:
        cmd += " -addext subjectAltName=" + ",".join(alt_names)
    subprocess.run(shlex.split(cmd), check=True, capture_output=True)
    return sv_csrq, sv_pkey


def gen_ssl_cert(
    folder: str,
    sv_csrq: str,
    ca_cert: str,
    ca_pkey: str,
    password: Optional[str] = None,
    old: bool = False,  # flag when using an old version (OpenSSL 1.x)
) -> str:
    """Sign a CSR into a certificate using a given CA."""
    # private method; pylint: disable=too-many-arguments
    sv_cert = os.path.join(folder, "server-cert.pem")
    cmd = (
        f"openssl x509 -req -sha256 -days 30 -in {sv_csrq} -out {sv_cert} "
        + f"-CA {ca_cert} -CAkey {ca_pkey} -CAcreateserial"
        + (" -copy_extensions=copy" if not old else "")
        + (f" -passin pass:{password}" if password else "")
    )
    subprocess.run(shlex.split(cmd), check=True, capture_output=True)
    return sv_cert
