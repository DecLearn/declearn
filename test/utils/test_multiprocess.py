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

"""Unit tests for 'declearn.utils.run_as_processes'."""

import time
from typing import Any, Dict, NoReturn, Tuple, Type

import pytest

from declearn.utils import run_as_processes


def sleep_routine(
    duration: float = 0.5,
) -> float:
    """Sleep for a given duration."""
    time.sleep(duration)
    return duration


def fail_routine(
    error_msg: str,
    duration: float = 0.5,
    error_cls: Type[Exception] = ValueError,
) -> NoReturn:
    """Sleep for a given duration, then raise with a given error message."""
    time.sleep(duration)
    raise error_cls(error_msg)


class TestRunAsProcesses:
    """Unit tests for 'declearn.utils.run_as_processes'."""

    # Tests with unproper inputs (priot to running any mp.Process).

    def test_unproper_specs_wrong_routine_type(self) -> None:
        """Test that a TypeError is raised on an unproper input type."""
        routine: Dict[Any, Any] = {}
        with pytest.raises(TypeError):
            run_as_processes(routine)  # type: ignore  # deliberate mistype

    def test_unproper_specs_wrong_routine_length(self) -> None:
        """Test that a TypeError is raised on an unproper input type."""
        routine = (sleep_routine,)
        with pytest.raises(TypeError):
            run_as_processes(routine)  # type: ignore  # deliberate mistype

    def test_unproper_specs_wrong_function_type(self) -> None:
        """Test that a TypeError is raised on an unproper input type."""
        routine: Tuple[None, Tuple] = (None, tuple())
        with pytest.raises(TypeError):
            run_as_processes(routine)  # type: ignore  # deliberate mistype

    def test_unproper_specs_wrong_args_kwargs_type(self) -> None:
        """Test that a TypeError is raised on an unproper input type."""
        routine = (sleep_routine, None)
        with pytest.raises(TypeError):
            run_as_processes(routine)  # type: ignore  # deliberate mistype

    def test_unproper_specs_wrong_args_kwargs_order(self) -> None:
        """Test that a TypeError is raised on an unproper input type."""
        routine = (sleep_routine, {"duration": 0.01}, tuple())  # type: ignore
        with pytest.raises(TypeError):
            run_as_processes(routine)  # type: ignore  # deliberate mistype

    # Tests with a single routine.

    def test_sleep_routine_with_args(self) -> None:
        """Test that running a single routine with args works properly."""
        success, outputs = run_as_processes((sleep_routine, (0.01,)))
        assert success
        assert isinstance(outputs, list) and len(outputs) == 1
        assert outputs[0] == 0.01

    def test_sleep_routine_with_kwargs(self) -> None:
        """Test that running a single routine with kwargs works properly."""
        success, outputs = run_as_processes(
            (sleep_routine, {"duration": (0.01)})
        )
        assert success
        assert isinstance(outputs, list) and len(outputs) == 1
        assert outputs[0] == 0.01

    def test_fail_routine_with_args_and_kwargs(self) -> None:
        """Test that running a single routine with args and kwargs works."""
        err_msg = "Triggered exception."
        success, outputs = run_as_processes(
            (fail_routine, (err_msg,), {"duration": 0.01})
        )
        assert not success
        assert isinstance(outputs, list) and len(outputs) == 1
        assert isinstance(outputs[0], RuntimeError)
        assert err_msg in str(outputs[0])
        assert ValueError.__name__ in str(outputs[0])

    # Tests with multiple routines.

    def test_multiple_sleep_routines(self) -> None:
        """Test that running a pair of parallel routines works properly."""
        success, outputs = run_as_processes(
            (sleep_routine, {"duration": 0.1}),
            (sleep_routine, (0.01,)),
        )
        assert success
        assert isinstance(outputs, list) and len(outputs) == 2
        assert outputs[0] == 0.1
        assert outputs[1] == 0.01

    def test_sleep_and_fail_with_autostop(self) -> None:
        """Test autostop=True, using a long routine and a failing one."""
        err_msg = "Mock exception."
        err_cls = AttributeError
        srt_time = time.time()
        success, outputs = run_as_processes(
            (sleep_routine, {"duration": 10.0}),
            (fail_routine, (err_msg, 0.01, err_cls)),
            auto_stop=True,
        )
        duration = time.time() - srt_time
        assert not success
        assert duration < 10.0  # sleep_routine should have been interrupted
        assert isinstance(outputs, list) and len(outputs) == 2
        assert isinstance(outputs[0], RuntimeError)
        assert "interrupted" in str(outputs[0])
        assert isinstance(outputs[1], RuntimeError)
        assert err_msg in str(outputs[1])
        assert err_cls.__name__ in str(outputs[1])

    def test_sleep_and_fail_without_autostop(self) -> None:
        """Test autostop=False, using a long routine and a failing one."""
        err_msg = "Mock exception."
        err_cls = AttributeError
        srt_time = time.time()
        success, outputs = run_as_processes(
            (sleep_routine, {"duration": 0.01}),
            (fail_routine, (err_msg, 0.001, err_cls)),
            auto_stop=False,
        )
        duration = time.time() - srt_time
        assert not success
        assert duration >= 0.01  # sleep_routine should have gone through
        assert isinstance(outputs, list) and len(outputs) == 2
        assert isinstance(outputs[0], float)
        assert outputs[0] == 0.01
        assert isinstance(outputs[1], RuntimeError)
        assert err_msg in str(outputs[1])
        assert err_cls.__name__ in str(outputs[1])
