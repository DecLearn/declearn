# coding: utf-8

"""Shared pytest fixtures for testing optmizer and plugins."""


import pytest

from declearn.test_utils import list_available_frameworks


@pytest.fixture(name="framework", params=list_available_frameworks())
def framework_fixture(request):
    """Fixture to provide with the name of a model framework."""
    return request.param
