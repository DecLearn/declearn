"""Unit tests for the construction of client samplers from specs"""

import pytest

from declearn.client_sampler import ClientSampler, CompositionClientSampler
from declearn.client_sampler.modules import (
    CompositionCriterion,
    ConstantCriterion,
    CriterionClientSampler,
    DefaultClientSampler,
    GradientNormCriterion,
    UniformClientSampler,
)

OPERATIONS = [
    "add",
    "+",
    "sub",
    "-",
    "mul",
    "*",
    "div",
    "truediv",
    "/",
    "radd",
    "rsub",
    "rmul",
    "rtruediv",
    "pow",
]


def test_from_specs_default():
    specs = {"strategy": "default"}

    sampler = ClientSampler.from_specs(**specs)
    assert isinstance(sampler, DefaultClientSampler)


def test_from_specs_unknown_strategy():
    """
    Test that we have an exception in case of a strategy value that
    does not exist
    """
    specs = {"strategy": "unknown"}
    with pytest.raises(ValueError):
        ClientSampler.from_specs(**specs)


def test_from_specs_uniform():
    specs = {
        "strategy": "uniform",
        "n_samples": 2,
        "seed": 42,
    }

    sampler = ClientSampler.from_specs(**specs)
    assert isinstance(sampler, UniformClientSampler)
    assert sampler.n_samples == 2
    assert sampler.seed == 42


def test_from_specs_uniform_wrong_param():
    specs = {
        "strategy": "uniform",
        "n_samples": 2,
        "wrong": True,
    }
    with pytest.raises(ValueError):
        ClientSampler.from_specs(**specs)


def test_from_specs_uniform_missing_param():
    specs = {
        "strategy": "uniform",
    }
    with pytest.raises(ValueError):
        ClientSampler.from_specs(**specs)


def test_from_specs_composition():
    specs = {
        "strategy": "composition",
        "samplers": [
            {
                "strategy": "uniform",
                "n_samples": 2,
                "seed": 42,
            },
            {
                "strategy": "default",
            },
        ],
    }
    sampler = ClientSampler.from_specs(**specs)
    assert isinstance(sampler, CompositionClientSampler)
    sampler1 = sampler.samplers[0]
    sampler2 = sampler.samplers[1]
    assert isinstance(sampler1, UniformClientSampler)
    assert sampler1.n_samples == 2
    assert sampler1.seed == 42
    assert isinstance(sampler2, DefaultClientSampler)


def test_from_specs_composition_with_objects():
    specs = {
        "strategy": "composition",
        "samplers": [
            UniformClientSampler(n_samples=2, seed=42),
            DefaultClientSampler(),
        ],
    }
    sampler = ClientSampler.from_specs(**specs)
    assert isinstance(sampler, CompositionClientSampler)
    sampler1 = sampler.samplers[0]
    sampler2 = sampler.samplers[1]
    assert isinstance(sampler1, UniformClientSampler)
    assert sampler1.n_samples == 2
    assert sampler1.seed == 42
    assert isinstance(sampler2, DefaultClientSampler)


def test_from_specs_criterion_grad_norm():
    specs = {
        "strategy": "criterion",
        "n_samples": 2,
        "criterion": {
            "name": "gradient_norm",
        },
        "missing_weights_policy": "priority",
    }
    sampler = ClientSampler.from_specs(**specs)
    assert isinstance(sampler, CriterionClientSampler)
    assert sampler.n_samples == 2
    assert sampler.missing_weights_policy == "priority"
    assert isinstance(sampler.criterion, GradientNormCriterion)


def test_from_specs_criterion_constant():
    specs = {
        "strategy": "criterion",
        "n_samples": 2,
        "criterion": {
            "name": "constant",
            "value": 1,
        },
        "missing_weights_policy": "priority",
    }
    sampler = ClientSampler.from_specs(**specs)
    assert isinstance(sampler, CriterionClientSampler)
    assert sampler.n_samples == 2
    assert sampler.missing_weights_policy == "priority"
    assert isinstance(sampler.criterion, ConstantCriterion)
    assert sampler.criterion.value == 1


def test_from_specs_criterion_unknown():
    """
    Test that we have an exception in case of a criterion name value that
    does not exist
    """
    specs = {
        "strategy": "criterion",
        "n_samples": 2,
        "criterion": {
            "name": "unknown",
        },
        "missing_weights_policy": "priority",
    }
    with pytest.raises(ValueError):
        ClientSampler.from_specs(**specs)


def test_from_specs_criterion_constant_wrong_param():
    specs = {
        "strategy": "criterion",
        "n_samples": 2,
        "criterion": {
            "name": "constant",
            "value": 1,
            "wrong": True,
        },
        "missing_weights_policy": "priority",
    }
    with pytest.raises(ValueError):
        ClientSampler.from_specs(**specs)


def test_from_specs_criterion_constant_missing_param():
    specs = {
        "strategy": "criterion",
        "n_samples": 2,
        "criterion": {
            "name": "constant",
        },
        "missing_weights_policy": "priority",
    }
    with pytest.raises(ValueError):
        ClientSampler.from_specs(**specs)


@pytest.mark.parametrize(
    "operation",
    OPERATIONS,
)
def test_from_specs_criterion_composition(operation: str):
    specs = {
        "strategy": "criterion",
        "n_samples": 2,
        "criterion": {
            "name": "composition",
            "operation": operation,
            "parents": [
                {
                    "name": "gradient_norm",
                },
                {
                    "name": "constant",
                    "value": 1,
                },
            ],
        },
        "missing_weights_policy": "priority",
    }
    sampler = ClientSampler.from_specs(**specs)
    assert isinstance(sampler, CriterionClientSampler)
    assert sampler.n_samples == 2
    assert sampler.missing_weights_policy == "priority"
    assert isinstance(sampler.criterion, CompositionCriterion)
    assert isinstance(sampler.criterion.parents[0], GradientNormCriterion)
    assert isinstance(sampler.criterion.parents[1], ConstantCriterion)
    assert sampler.criterion.parents[1].value == 1
