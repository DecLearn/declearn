# coding: utf-8

# Copyright 2023 Inria (Institut National de Recherche en Informatique
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

"""Miscellaneous private backend utils used in model code."""

from typing import Set


__all__ = [
    "raise_on_stringsets_mismatch",
]


def raise_on_stringsets_mismatch(
    received: Set[str],
    expected: Set[str],
    context: str = "expected",
) -> None:
    """Raise a verbose KeyError if two sets of strings do not match.

    Parameters
    ----------
    received: set[str]
        Received set of string values.
    expected: set[str]
        Expected set of string values.
    context: str, default="expected"
        String piece used in the raised exception's description to
        designate the `expected` names.

    Raises
    ------
    KeyError:
        In case `received != expected`.
        Verbose about the missing and/or unexpected `received` keys.
    """
    if received != expected:
        missing = expected.difference(received)
        unexpct = received.difference(expected)
        raise KeyError(
            f"Mismatch between input and {context} names:\n"
            + f"Missing key(s) in inputs: {missing}\n" * bool(missing)
            + f"Unexpected key(s) in inputs: {unexpct}\n" * bool(unexpct)
        )
