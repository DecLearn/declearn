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

"""Shared pytest configuration code for the test suite."""

import pytest


def pytest_addoption(parser) -> None:
    """Add some custom options to the pytest commandline."""
    parser.addoption(
        "--fulltest",
        action="store_true",
        default=False,
        help="--fulltest: run all test scenarios in 'test_main.py'",
    )
    parser.addoption(
        "--cpu-only",
        action="store_true",
        default=False,
        help="--cpu-only: disable the use of GPU devices in tests",
    )


@pytest.fixture(name="fulltest")
def fulltest_fixture(request) -> bool:
    """Gather the '--fulltest' option's value."""
    return bool(request.config.getoption("--fulltest"))


@pytest.fixture(name="cpu_only")
def cpu_only_fixture(request) -> bool:
    """Gather the '--cpu-only' option's value."""
    return bool(request.config.getoption("--cpu-only"))
