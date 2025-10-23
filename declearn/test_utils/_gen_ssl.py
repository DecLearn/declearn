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

"""Utils to automate self-signing-based SSL certificates generation."""

import datetime
import ipaddress
import os
from typing import Collection, Optional, Tuple

import cryptography.hazmat.primitives.asymmetric.rsa
import cryptography.hazmat.primitives.hashes
import cryptography.hazmat.primitives.serialization as crypto_serialization
from cryptography import x509
from cryptography.x509.oid import NameOID

__all__ = [
    "generate_ssl_certificates",
]


# pylint: disable-next=too-many-positional-arguments
def generate_ssl_certificates(  # noqa: PLR0913
    folder: str = ".",
    c_name: str = "localhost",
    password: Optional[str] = None,
    alt_ips: Optional[Collection[str]] = None,
    alt_dns: Optional[Collection[str]] = None,
    duration: int = 30,
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
    duration: int, default=30
        Validity duration for both the CA and server certificates.

    Returns
    -------
    ca_cert: str
        Path to the client-required CA certificate PEM file.
    sv_cert: str
        Path to the server's certificate PEM file.
    sv_pkey: str
        Path to the server's private key PEM file.
    """
    # arguments serve modularity; pylint: disable=too-many-arguments
    # Generate a self-signed root CA.
    ca_cert, ca_pkey = gen_ssl_ca(folder, password, duration)
    # Generate a server CSR and a private key.
    sv_csrq, sv_pkey = gen_ssl_csr(folder, c_name, alt_ips, alt_dns, password)
    # Sign the CSR into a server certificate using the root CA.
    sv_cert = gen_ssl_cert(
        folder, sv_csrq, ca_cert, ca_pkey, password, duration
    )
    # Return paths that are used by declearn network-communication endpoints.
    return ca_cert, sv_cert, sv_pkey


def generate_private_key(
    path: str,
    password: Optional[str],
    key_size: int = 4096,
) -> None:
    """Generate a private RSA key.

    Parameters
    ----------
    path:
        Path to the output PEM file.
    password:
        Optional password to secure the created PEM file.
    key_size:
        Size of the generated RSA key.
    """
    key = cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )
    if password is None:
        encryption_algorithm: crypto_serialization.KeySerializationEncryption = crypto_serialization.NoEncryption()
    else:
        encryption_algorithm = crypto_serialization.BestAvailableEncryption(
            password.encode("utf-8")
        )
    key_bytes = key.private_bytes(
        encoding=crypto_serialization.Encoding.PEM,
        format=crypto_serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=encryption_algorithm,
    )
    with open(path, "wb") as file:
        file.write(key_bytes)


def load_private_rsa_key(
    path: str,
    password: Optional[str],
) -> cryptography.hazmat.primitives.asymmetric.rsa.RSAPrivateKey:
    """Load a private rsa key from a PEM file.

    Parameters
    ----------
    path:
        Path to the PEM file where the RSA key is stored.
    password:
        Optional password to decrypt the PEM file.
    """
    with open(path, "rb") as file:
        data = file.read()
    passbytes = None if password is None else password.encode("utf-8")
    key = crypto_serialization.load_pem_private_key(data, password=passbytes)
    if not isinstance(
        key, cryptography.hazmat.primitives.asymmetric.rsa.RSAPrivateKey
    ):
        raise TypeError(f"File {path} does not hold a private RSA key.")
    return key


def gen_ssl_ca(
    folder: str,
    password: Optional[str] = None,
    duration: int = 30,
) -> Tuple[str, str]:
    """Generate a self-signed CA certificate and its private key.

    Parameters
    ----------
    folder:
        Path to the root folder where to output created files.
    password:
        Optional password to encrypt the private key file.

    Returns
    -------
    ca_path:
        Path to the created CA PEM file.
    key_path:
        Path to the created private RSA key PEM file.
    duration:
        Validity duration of the created certificate, in days.
    """
    # Generate a private key and load it in memory.
    ca_pkey = os.path.join(folder, "ca-pkey.pem")
    generate_private_key(ca_pkey, password=password)
    key = load_private_rsa_key(ca_pkey, password=password)
    # Generate the self-signed CA certificate.
    identifiers = [
        x509.NameAttribute(NameOID.COUNTRY_NAME, "FR"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Lille"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Inria"),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Magnet"),
        x509.NameAttribute(NameOID.COMMON_NAME, "SelfSignedCA"),
    ]
    subject_name = x509.Name(identifiers)
    today = datetime.datetime.now(datetime.timezone.utc)
    cert_builder = x509.CertificateBuilder(
        subject_name=subject_name,
        issuer_name=subject_name,
        public_key=key.public_key(),
        serial_number=x509.random_serial_number(),
        not_valid_before=today,
        not_valid_after=today + datetime.timedelta(days=duration),
    )
    cert_builder = cert_builder.add_extension(
        x509.BasicConstraints(ca=True, path_length=None), critical=True
    )
    cert = cert_builder.sign(
        key, cryptography.hazmat.primitives.hashes.SHA256()
    )
    # Export the certificate to a PEM file.
    ca_cert = os.path.join(folder, "ca-cert.pem")
    cert_bytes = cert.public_bytes(crypto_serialization.Encoding.PEM)
    with open(ca_cert, "wb") as file:
        file.write(cert_bytes)
    # Return paths to the certificate and its private key.
    return ca_cert, ca_pkey


def gen_ssl_csr(
    folder: str,
    c_name: str,
    alt_ips: Optional[Collection[str]] = None,
    alt_dns: Optional[Collection[str]] = None,
    password: Optional[str] = None,
) -> Tuple[str, str]:
    """Generate a CSR (certificate signing request) and its private key.

    Parameters
    ----------
    folder:
        Path to the root folder where to output created files.
    c_name:
        Common Name to indicate as part of the CSR.
    alt_ips:
        Optional list of alternative IP adresses for the CSR to cover.
    alt_dns:
        Optional list of alternative domain names for the CSR to cover.

    Returns
    -------
    csr_path:
        Path to the created CSR PEM file.
    key_path:
        Path to the created private RSA key PEM file.
    """
    # Generate a private RSA key and load it in memory.
    sv_pkey = os.path.join(folder, "server-pkey.pem")
    generate_private_key(sv_pkey, password=password)
    key = load_private_rsa_key(sv_pkey, password=password)
    # Set up the CSR.
    identifiers = [
        x509.NameAttribute(NameOID.COUNTRY_NAME, "FR"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Lille"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Inria"),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Magnet"),
        x509.NameAttribute(NameOID.COMMON_NAME, c_name),
    ]
    csr_builder = x509.CertificateSigningRequestBuilder(
        subject_name=x509.Name(identifiers),
    )
    # Optionally add alternative IPs and/or DNS.
    alt_names = [
        *[x509.IPAddress(ipaddress.ip_address(ip)) for ip in (alt_ips or [])],
        *[x509.DNSName(dns) for dns in (alt_dns or [])],
    ]
    if alt_names:
        csr_builder = csr_builder.add_extension(
            x509.SubjectAlternativeName(alt_names),
            critical=True,
        )
    # Sign the CSR using the private key, then export it to disk.
    csr = csr_builder.sign(key, cryptography.hazmat.primitives.hashes.SHA256())
    sv_csrq = os.path.join(folder, "server-csrq.pem")
    csr_bytes = csr.public_bytes(crypto_serialization.Encoding.PEM)
    with open(sv_csrq, "wb") as file:
        file.write(csr_bytes)
    # Retun paths to the private key and CSR files.
    return sv_csrq, sv_pkey


# pylint: disable-next=too-many-positional-arguments
def gen_ssl_cert(  # noqa: PLR0913
    folder: str,
    sv_csrq: str,
    ca_cert: str,
    ca_pkey: str,
    password: Optional[str] = None,
    duration: int = 30,
) -> str:
    """Sign a CSR into a certificate using a given CA.

    Parameters
    ----------
    folder:
        Root folder where to output the created file.
    sv_csrq:
        Path to the CSR file that needs signing.
    ca_cert:
        Path to the CA file to use for signing.
    ca_pkey:
        Path to the private key of the CA.
    password:
        Optional password to decrypt the CA private key.
    duration:
        Validity duration of the created certificate, in days.

    Returns
    -------
    cert_path:
        Path to the created certificate PEM file.
    """
    # backend function; pylint: disable=too-many-arguments
    # Load the CSR, the CA cert and its private key.
    with open(sv_csrq, "rb") as file:
        csr = x509.load_pem_x509_csr(file.read())
    with open(ca_cert, "rb") as file:
        cac = x509.load_pem_x509_certificate(file.read())
    key = load_private_rsa_key(ca_pkey, password=password)
    # Set up and sign the certificate.
    today = datetime.datetime.now(datetime.timezone.utc)
    cert = x509.CertificateBuilder(
        subject_name=csr.subject,
        issuer_name=cac.subject,
        public_key=csr.public_key(),
        serial_number=x509.random_serial_number(),
        not_valid_before=today,
        not_valid_after=today + datetime.timedelta(days=duration),
        extensions=list(csr.extensions),
    ).sign(key, cryptography.hazmat.primitives.hashes.SHA256())
    # Export the certificate to a PEM file and return its path.
    sv_cert = os.path.join(folder, "server-cert.pem")
    cert_bytes = cert.public_bytes(crypto_serialization.Encoding.PEM)
    with open(sv_cert, "wb") as file:
        file.write(cert_bytes)
    return sv_cert
