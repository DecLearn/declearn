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

"""Unit tests for Joye-Libert setup routines."""

import os
from typing import Any, Dict

from declearn.secagg.masking import (
    MaskingDecrypter,
    MaskingEncrypter,
    MaskingSecaggConfigClient,
    MaskingSecaggConfigServer,
)
from declearn.secagg.masking.messages import MaskingSecaggSetupQuery
from declearn.test_utils import make_importable

with make_importable(os.path.dirname(os.path.abspath(__file__))):
    from secagg_setup_testing import SecaggSetupTestCase


class TestMaskingSecaggSetup(SecaggSetupTestCase):
    """Unit tests for Masking-based SecAgg setup and config classes."""

    decrypter_cls = MaskingDecrypter
    encrypter_cls = MaskingEncrypter
    client_config_cls = MaskingSecaggConfigClient
    server_config_cls = MaskingSecaggConfigServer
    setup_msg_cls = MaskingSecaggSetupQuery

    def get_server_hyper_parameters(
        self,
    ) -> Dict[str, Any]:
        """Return server-side arbitraty config hyper-parameters."""
        kwargs = super().get_server_hyper_parameters()
        kwargs["bitsize"] = 32  # default 16 bits are rather low for masking
        return kwargs

    def assert_decrypter_validity(
        self,
        decrypter: MaskingDecrypter,
        **kwargs: Any,
    ) -> None:
        # Verify decrypter type and quantization clipval.
        super().assert_decrypter_validity(decrypter, **kwargs)
        # Verify quantization integer range.
        max_irange = 2 ** kwargs["bitsize"]
        assert decrypter.quantizer.int_range * kwargs["n_clients"] < max_irange
