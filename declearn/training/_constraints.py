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

"""Minimal API to design and enforce computational effort constraints."""

import time
from typing import Dict, Optional

__all__ = [
    "Constraint",
    "ConstraintSet",
    "TimeoutConstraint",
]


class Constraint:
    """Base class to implement effort constraints.

    This class defines a generic API and implements count-based constraints.

    Usage
    -----
    * When instantiated, a base Constraint records a `self.value`
      counter, initialized based on the `start` parameter.
    * On each `self.increment()` call, the `self.value` attribute
      is incremented by 1.
    * The constraint is `saturated` when `self.value > self.limit`.
    """

    def __init__(
        self,
        limit: Optional[float],
        start: float = 0.0,
        name: str = "constraint",
    ) -> None:
        """Instantiate a count-based constraint.

        Parameters
        ----------
        limit: float or None
            Value beyond which the constraint is saturated.
            If None, set to Inf, making this a counter.
        start: float, default=0.
            Start value of the counter.
            This value goes up by 1 on each `self.increment` call.
        name: str, default="constraint"
            Name of the constraint.
        """
        self.limit = limit or float("inf")
        self.value = start
        self.name = name

    def increment(
        self,
    ) -> None:
        """Update `self.value`, incrementing it by 1."""
        self.value += 1

    @property
    def saturated(
        self,
    ) -> bool:
        """Return whether the constraint is saturated."""
        return self.value >= self.limit


class TimeoutConstraint(Constraint):
    """Class implementing a simple time-based constraint.

    Usage
    -----
    * When instantiated, a TimeoutConstraint records a reference
      time (that at instantiation, optionally adjusted based on
      the `start` parameter) under the `self.start` attribute.
    * On each `self.increment()` call, the `self.value` attribute
      is updated to the difference between the current time and
      the `self.start` attribute.
    * The constraint is `saturated` when `self.value > self.limit`.
    """

    def __init__(
        self,
        limit: Optional[float],
        start: float = 0.0,
        name: str = "timeout",
    ) -> None:
        """Instantiate a time-based constraint.

        Parameters
        ----------
        limit: float or None
            Value beyond which the constraint is saturated.
            If None, set to Inf, making this a counter.
        start: float, default=0.
            Start duration to substract from the current time
            to set up the reference starting time.
        name: str, default="timeout"
            Name of the constraint.
        """
        super().__init__(limit, start, name)
        self.start = time.time() - start

    def increment(
        self,
    ) -> None:
        """Update `self.value`, storing time passed since `self.start`."""
        self.value = time.time() - self.start


class ConstraintSet:
    """Utility class to wrap sets of Constraint instances."""

    def __init__(
        self,
        *constraints: Constraint,
    ) -> None:
        """Wrap an ensemble of Constraint objects."""
        self.constraints = constraints

    def increment(
        self,
    ) -> None:
        """Increment each and every wrapped constraint."""
        for constraint in self.constraints:
            constraint.increment()

    @property
    def saturated(
        self,
    ) -> bool:
        """Return whether any wrapped constraint is saturated."""
        return any(c.saturated for c in self.constraints)

    def get_values(
        self,
    ) -> Dict[str, float]:
        """Return the wrapped constraints' current values, as a dict.

        Returns
        -------
        values: dict[str, float]
            {constraint.name: constraint.value} dictionary.
            If multiple constraints have the same name, suffixes
            will be appended in order to disambiguate them.
        """
        values: Dict[str, float] = {}
        for constraint in self.constraints:
            name = constraint.name
            idx = 0
            while name in values:
                name = f"{constraint.name}.{idx}"
                idx += 1
            values[name] = constraint.value
        return values
