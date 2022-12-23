# coding: utf-8

"""Shared pytest fixtures for testing optmizer and plugins."""


import pytest

from declearn.test_utils import Frameworks


@pytest.fixture(name="framework", params=Frameworks)
def framework_fixture(request):
    """Fixture to provide with the name of a model framework."""
    return request.param
