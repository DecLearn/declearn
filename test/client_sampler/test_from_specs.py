# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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

"""Unit tests for the construction of client samplers from specs / configs."""

import pytest

from declearn.client_sampler import (
    ClientSamplerConfig,
    CompositionClientSampler,
    CriterionClientSampler,
    DefaultClientSampler,
    UniformClientSampler,
    instantiate_client_sampler,
)
from declearn.client_sampler.criterion import (
    CompositionCriterion,
    ConstantCriterion,
    GradientNormCriterion,
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
    "pow",
]

## Tests of construction from specs as dictionnaries


def test_from_specs_default():
    specs = {"strategy": "default"}

    sampler = instantiate_client_sampler(**specs)
    assert isinstance(sampler, DefaultClientSampler)


def test_from_specs_unknown_strategy():
    """
    Test that we have an exception in case of a strategy value that
    does not exist
    """
    specs = {"strategy": "unknown"}
    with pytest.raises(ValueError):
        instantiate_client_sampler(**specs)


def test_from_specs_uniform():
    specs = {
        "strategy": "uniform",
        "n_samples": 2,
        "seed": 42,
    }

    sampler = instantiate_client_sampler(**specs)
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
        instantiate_client_sampler(**specs)


def test_from_specs_uniform_missing_param():
    specs = {
        "strategy": "uniform",
    }
    with pytest.raises(ValueError):
        instantiate_client_sampler(**specs)


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
    sampler = instantiate_client_sampler(**specs)
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
    sampler = instantiate_client_sampler(**specs)
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
        "missing_scores_policy": "priority",
    }
    sampler = instantiate_client_sampler(**specs)
    assert isinstance(sampler, CriterionClientSampler)
    assert sampler.n_samples == 2
    assert sampler.missing_scores_policy == "priority"
    assert isinstance(sampler.criterion, GradientNormCriterion)


def test_from_specs_criterion_constant():
    specs = {
        "strategy": "criterion",
        "n_samples": 2,
        "criterion": {
            "name": "constant",
            "value": 1,
        },
        "missing_scores_policy": "priority",
    }
    sampler = instantiate_client_sampler(**specs)
    assert isinstance(sampler, CriterionClientSampler)
    assert sampler.n_samples == 2
    assert sampler.missing_scores_policy == "priority"
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
        "missing_scores_policy": "priority",
    }
    with pytest.raises(ValueError):
        instantiate_client_sampler(**specs)


def test_from_specs_criterion_constant_wrong_param():
    specs = {
        "strategy": "criterion",
        "n_samples": 2,
        "criterion": {
            "name": "constant",
            "value": 1,
            "wrong": True,
        },
        "missing_scores_policy": "priority",
    }
    with pytest.raises(ValueError):
        instantiate_client_sampler(**specs)


def test_from_specs_criterion_constant_missing_param():
    specs = {
        "strategy": "criterion",
        "n_samples": 2,
        "criterion": {
            "name": "constant",
        },
        "missing_scores_policy": "priority",
    }
    with pytest.raises(ValueError):
        instantiate_client_sampler(**specs)


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
        "missing_scores_policy": "priority",
    }
    sampler = instantiate_client_sampler(**specs)
    assert isinstance(sampler, CriterionClientSampler)
    assert sampler.n_samples == 2
    assert sampler.missing_scores_policy == "priority"
    assert isinstance(sampler.criterion, CompositionCriterion)
    assert isinstance(sampler.criterion.parents[0], GradientNormCriterion)
    assert isinstance(sampler.criterion.parents[1], ConstantCriterion)
    assert sampler.criterion.parents[1].value == 1


def test_from_specs_criterion_composition_wrong_operation():
    specs = {
        "strategy": "criterion",
        "n_samples": 2,
        "criterion": {
            "name": "composition",
            "operation": "wrong",
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
        "missing_scores_policy": "priority",
    }
    with pytest.raises(ValueError):
        instantiate_client_sampler(**specs)


def test_from_specs_criterion_composition_wrong_operation_type():
    specs = {
        "strategy": "criterion",
        "n_samples": 2,
        "criterion": {
            "name": "composition",
            "operation": True,
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
        "missing_scores_policy": "priority",
    }
    with pytest.raises(ValueError):
        instantiate_client_sampler(**specs)


## Tests of construction from specs in TomlConfig and TOML files


def test_from_toml_config_simple(tmp_path):
    toml_content = """
    [client_sampler]
    strategy = "uniform"

    [client_sampler.params]
    n_samples = 2
    seed = 42
    max_retries = 3
    """

    toml_file = tmp_path / "config.toml"
    toml_file.write_text(toml_content)

    sampler_config = ClientSamplerConfig.from_toml(
        toml_file, False, "client_sampler"
    )
    sampler = sampler_config.build()

    assert isinstance(sampler, UniformClientSampler)
    assert sampler.n_samples == 2
    assert sampler.seed == 42
    assert sampler.max_retries == 3


def test_from_toml_config_wrong(tmp_path):
    toml_content = """
    [client_sampler]
    strategy = "uniform"

    [client_sampler.params]
    n_samples = 2
    wrong = 1
    """

    toml_file = tmp_path / "config.toml"
    toml_file.write_text(toml_content)

    sampler_config = ClientSamplerConfig.from_toml(
        toml_file, False, "client_sampler"
    )
    with pytest.raises(ValueError):
        sampler_config.build()


def test_from_toml_config_complex(tmp_path):
    toml_content = """
    [client_sampler]
    strategy = "criterion"

    [client_sampler.params]
    n_samples = 2
    missing_scores_policy = "priority"

    [client_sampler.params.criterion]
    name = "composition"
    operation = "add"

    [[client_sampler.params.criterion.parents]]
    name = "gradient_norm"

    [[client_sampler.params.criterion.parents]]
    name = "constant"
    value = 1
    """

    toml_file = tmp_path / "config.toml"
    toml_file.write_text(toml_content)

    config = ClientSamplerConfig.from_toml(toml_file, False, "client_sampler")
    sampler = config.build()

    assert isinstance(sampler, CriterionClientSampler)
    assert sampler.n_samples == 2
    assert sampler.missing_scores_policy == "priority"
    assert isinstance(sampler.criterion, CompositionCriterion)
    assert isinstance(sampler.criterion.parents[0], GradientNormCriterion)
    assert isinstance(sampler.criterion.parents[1], ConstantCriterion)
    assert sampler.criterion.parents[1].value == 1
