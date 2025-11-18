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

"""Shared pytest fixtures for testing Vector classes."""

import pytest

from declearn.utils import set_device_policy


@pytest.fixture(autouse=True)
def disable_gpu():
    """Ensure 'Vector' unit tests run on CPU only.

    Note: GPU is tested functionnally as part of 'Model' unit tests.
    It may be interesting in the future to add dedicated tests for
    GPU support (notably as to operations on Vectors that are placed
    on distinct devices).
    """
    set_device_policy(gpu=False)
