# coding: utf-8

"""Shared pytest configuration code for the test suite."""

import pytest


def pytest_addoption(parser) -> None:  # type: ignore
    """Add a '--fulltest' option to the pytest commandline."""
    parser.addoption(
        "--fulltest", action="store_true", default=False,
        help="--fulltest: run all test scenarios in 'test_main.py'"
    )


@pytest.fixture(name="fulltest")
def fulltest_fixture(request) -> bool:  # type: ignore
    """Gather the '--fulltest' option's value."""
    return bool(request.config.getoption("--fulltest"))
