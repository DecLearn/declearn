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

"""Unit tests for Scheduler classes."""

import warnings
from copy import deepcopy
from typing import List

import pytest

from declearn.optimizer import list_rate_schedulers
from declearn.optimizer.schedulers import (
    CosineAnnealing,
    CosineAnnealingWarmRestarts,
    CyclicExpRange,
    CyclicTriangular,
    ExponentialDecay,
    InverseScaling,
    LinearDecay,
    PiecewiseDecay,
    PolynomialDecay,
    Scheduler,
    Warmup,
    WarmupRounds,
)
from declearn.test_utils import (
    assert_dict_equal,
    assert_json_serializable_dict,
)

SCHEDULERS: List[Scheduler] = [
    CosineAnnealing(0.001, max_lr=0.01, duration=10, step_level=False),
    CosineAnnealingWarmRestarts(0.001, max_lr=0.01, period=100, t_mult=0.5),
    CyclicExpRange(0.001, max_lr=0.01, stepsize=30, decay=0.9),
    CyclicTriangular(0.001, max_lr=0.01, stepsize=30, decay=True),
    ExponentialDecay(0.001, rate=0.1),
    InverseScaling(0.001, rate=0.5),
    LinearDecay(0.001, rate=0.0001, step_level=False),
    PiecewiseDecay(0.001, rate=0.5, step_size=2),
    PolynomialDecay(0.001, power=3, limit=10, step_level=False),
    Warmup(0.001, warmup=100),
    WarmupRounds(0.001, warmup=2),
]
SCHEDULERS_DICT = {scheduler.name: scheduler for scheduler in SCHEDULERS}
SCHEDULERS_DICT["warmup-decay"] = Warmup(LinearDecay(0.001, 0.1), warmup=100)
SCHEDULERS_DICT["warmup-rounds-decay"] = WarmupRounds(
    LinearDecay(0.001, 0.1), warmup=2
)


@pytest.fixture(name="scheduler")
def scheduler_fixture(
    name: str,
) -> Scheduler:
    """Scheduler-providing fixture."""
    return deepcopy(SCHEDULERS_DICT[name])


@pytest.mark.parametrize("name", list(SCHEDULERS_DICT))
class TestScheduler:
    """Unit tests for Scheduler subclasses.

    This class implements generic unit tests that are applied
    to a number of pre-parametrized 'Scheduler' instances.

    Note that it **does not test computations correctness**,
    but merely assesses whether the provided instance can be
    properly manipulated as per the defined Scheduler API.
    """

    def test_compute_value(
        self,
        scheduler: Scheduler,
    ) -> None:
        """Test that 'compute_value' returns time-based float values."""
        # Compute values at various steps and rounds.
        val_a = scheduler.compute_value(step=0, round_=0)
        val_b = scheduler.compute_value(step=1, round_=0)
        val_c = scheduler.compute_value(step=2, round_=1)
        # Assert that all values are float and differ.
        assert isinstance(val_a, float)
        assert isinstance(val_b, float)
        assert isinstance(val_c, float)
        assert (val_a != val_b) or (val_a != val_c)

    def test_get_next_rate(
        self,
        scheduler: Scheduler,
    ) -> None:
        """Test that 'get_next_rate' properly increments 'steps' counter."""
        scheduler.on_round_start()  # start round 0
        rate_0 = scheduler.get_next_rate()
        rate_1 = scheduler.get_next_rate()
        assert scheduler.steps == 2
        assert scheduler.rounds == 0
        assert rate_0 == scheduler.compute_value(step=0, round_=0)
        assert rate_1 == scheduler.compute_value(step=1, round_=0)

    def test_on_round_start(
        self,
        scheduler: Scheduler,
    ) -> None:
        """Test that 'on_round_start' properly increments 'rounds' counter."""
        scheduler.on_round_start()
        rate_0 = scheduler.get_next_rate()
        scheduler.on_round_start()
        rate_1 = scheduler.get_next_rate()
        assert scheduler.steps == 2
        assert scheduler.rounds == 1
        assert rate_0 == scheduler.compute_value(step=0, round_=0)
        assert rate_1 == scheduler.compute_value(step=1, round_=1)

    def test_get_config(
        self,
        scheduler: Scheduler,
    ) -> None:
        """Test that 'get_config' returns a JSON-serializable dict."""
        config = scheduler.get_config()
        assert_json_serializable_dict(config)

    def test_from_config(
        self,
        scheduler: Scheduler,
    ) -> None:
        """Test that a Scheduler can be recreated 'from_config'."""
        config = scheduler.get_config()
        schedul_b = type(scheduler).from_config(config)
        assert isinstance(schedul_b, type(scheduler))
        conf_b = schedul_b.get_config()
        assert_dict_equal(config, conf_b)

    def test_from_specs(
        self,
        scheduler: Scheduler,
    ) -> None:
        """Test that a Scheduler can be recreated 'from_specs'."""
        config = scheduler.get_config()
        schedul_b = Scheduler.from_specs(name=scheduler.name, config=config)
        assert isinstance(schedul_b, type(scheduler))
        conf_b = schedul_b.get_config()
        assert_dict_equal(config, conf_b)

    def test_get_state(
        self,
        scheduler: Scheduler,
    ) -> None:
        """Test that 'get_state' returns a JSON-serializable dict."""
        state = scheduler.get_state()
        assert_json_serializable_dict(state)

    def test_set_state(
        self,
        scheduler: Scheduler,
    ) -> None:
        """Test that 'set_state' works appropriately."""
        # Get the state then rate at steps (0, round_=0) and (120, round_=2).
        scheduler.on_round_start()
        state_0 = scheduler.get_state()
        rate_0 = scheduler.get_next_rate()
        scheduler.on_round_start()
        for _ in range(118):
            scheduler.get_next_rate()
        scheduler.on_round_start()
        state_120 = scheduler.get_state()
        rate_120 = scheduler.get_next_rate()
        # Set state and test that rates are re-computed identically.
        scheduler.set_state(state_0)
        assert scheduler.get_next_rate() == rate_0
        scheduler.set_state(state_120)
        assert scheduler.get_next_rate() == rate_120


def test_list_rate_schedulers():
    """Test that 'list_rate_schedulers' works properly."""
    schedulers = list_rate_schedulers()
    for name, cls in schedulers.items():
        assert issubclass(cls, Scheduler)
        assert cls.name == name
        if cls.name not in SCHEDULERS_DICT and cls.__module__.startswith(
            "declearn."
        ):
            warnings.warn(
                f"Registered Scheduler class '{cls}' is not covered by tests.",
                stacklevel=2,
            )
