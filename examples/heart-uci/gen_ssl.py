"""Script to generate self-signed SSL certificates for the demo."""

import os

from declearn.test_utils import generate_ssl_certificates


if __name__ == "__main__":
    FILEDIR = os.path.dirname(os.path.abspath(__file__))
    generate_ssl_certificates(FILEDIR)
