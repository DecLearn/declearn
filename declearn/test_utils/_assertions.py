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

"""Custom "assert" functions commonly used in declearn tests."""

import json
from collections.abc import Generator, Sequence
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from numpy.testing import assert_array_equal

from declearn.test_utils._convert import to_numpy
from declearn.utils import json_pack, json_unpack

__all__ = [
    "assert_dict_equal",
    "assert_json_serializable_dict",
    "assert_list_equal",
    "assert_batch_equal",
]


def assert_json_serializable_dict(sdict: Dict[str, Any]) -> None:
    """Assert that an input is JSON-serializable using declearn hooks.

    This function tries to dump the input dict into a JSON string,
    then to reload it. It does so using `declearn.utils.json_pack`
    and `json_unpack` functions to extend JSON (en|de)coding. It
    also asserts that the recovered dict is similar to the initial
    one, using the `assert_dict_equal` util (which tolerates list-
    to-tuple conversions induced by JSON).

    Parameters
    ----------
    sdict: Dict[str, Any]
        Dictionary, the JSON-serializability of which to assert.

    Raises
    ------
    AssertionError
        If `sdict` or the JSON-reloaded object is not a dict, or
        if the latter has different keys and/or values compared
        to the former.
    Exception
        Other exceptions may be raised if the JSON encoding (or
        decoding) operation goes wrong.
    """
    assert isinstance(sdict, dict)
    dump = json.dumps(sdict, default=json_pack)
    load = json.loads(dump, object_hook=json_unpack)
    assert isinstance(load, dict)
    assert_dict_equal(load, sdict)


def assert_dict_equal(
    dict_a: Dict[str, Any],
    dict_b: Dict[str, Any],
    strict_tuple: bool = False,
    np_tolerance: Optional[float] = None,
) -> None:
    """Assert that two (possibly nested) dicts are equal.

    This function is a more complex equivalent of `assert dict_a == dict_b`
    that enables comparing numpy array values, and optionally accepting to
    cast tuples as lists rather than assert that a tuple and a list are not
    equal in any case (even when their contents are the same).

    Parameters
    ----------
    dict_a: dict
        First dict to compare.
    dict_b: dict
        Second dict to compare.
    strict_tuple: bool, default=False
        Whether to cast tuples to list prior to comparing them
        (enabling some tuple-list type differences between the
        two compared dicts).
    np_tolerance: float or none, default=None
        Optional absolute tolerance to numpy arrays or float values'
        differences (use `np.allclose(a, b, rtol=0, atol=np_tolerance)`).

    Raises
    ------
    AssertionError
        If the two dicts are not equal.
    """
    assert dict_a.keys() == dict_b.keys()
    for key, val_a in dict_a.items():
        val_b = dict_b[key]
        assert_values_equal(val_a, val_b, strict_tuple, np_tolerance)


def assert_list_equal(
    list_a: Union[Tuple[Any], List[Any]],
    list_b: Union[Tuple[Any], List[Any]],
    strict_tuple: bool = False,
    np_tolerance: Optional[float] = None,
) -> None:
    """Assert that two (possibly nested) lists are equal.

    This function is a more complex equivalent of `assert list_a == list_b`
    that enables comparing numpy array values, and optionally accepting to
    cast tuples as lists rather than assert that a tuple and a list are not
    equal in any case (even when their contents are the same).

    Parameters
    ----------
    list_a: list
        First list to compare.
    list_b: list
        Second list to compare.
    strict_tuple: bool, default=False
        Whether to cast tuples to list prior to comparing them
        (enabling some tuple-list type differences between the
        two compared lists).
    np_tolerance: float or none, default=None
        Optional absolute tolerance to numpy arrays or float values'
        differences (use `np.allclose(a, b, rtol=0, atol=np_tolerance)`).

    Raises
    ------
    AssertionError
        If the two lists are not equal.
    """
    assert len(list_a) == len(list_b)
    for val_a, val_b in zip(list_a, list_b, strict=False):
        assert_values_equal(val_a, val_b, strict_tuple, np_tolerance)


def assert_values_equal(
    val_a: Any,
    val_b: Any,
    strict_tuple: bool = False,
    np_tolerance: Optional[float] = None,
) -> None:
    """Assert that two variables are equal

    This function is a more complex equivalent of `assert val_a == val_b`
    that enables comparing numpy array values, and optionally accepting to
    cast tuples as lists rather than assert that a tuple and a list are not
    equal in any case (even when their contents are the same). It relies on
    recursively comparing the elements of dict and list inputs.

    Parameters
    ----------
    val_a: list
        First variable to compare.
    val_b: list
        Second variable to compare.
    strict_tuple: bool, default=False
        Whether to cast tuples to list prior to comparing them
        (enabling some tuple-list type differences between the
        two compared values).
    np_tolerance: float or none, default=None
        Optional absolute tolerance to numpy arrays or float values'
        differences (use `np.allclose(a, b, rtol=0, atol=np_tolerance)`).

    Raises
    ------
    AssertionError
        If the two lists are not equal.
    """
    if isinstance(val_a, dict):
        assert isinstance(val_b, dict)
        assert_dict_equal(val_a, val_b, strict_tuple)
    elif isinstance(val_a, np.ndarray):
        assert isinstance(val_b, np.ndarray)
        assert val_a.shape == val_b.shape
        if np_tolerance:
            assert np.allclose(val_a, val_b, atol=np_tolerance, rtol=0.0)
        else:
            assert np.all(val_a == val_b)
    elif isinstance(val_a, (tuple, list)):
        if strict_tuple:
            assert isinstance(val_a, type(val_b))
        else:
            assert isinstance(val_a, (tuple, list))
        assert_list_equal(val_a, val_b, strict_tuple)
    elif isinstance(val_a, float) and np_tolerance:
        assert np.allclose(val_a, val_b, atol=np_tolerance, rtol=0.0)
    else:
        assert val_a == val_b


def flatten_and_assert(
    nested_x: Sequence,
    nested_y: Sequence,
    unpack_types: Tuple[Type[Any], ...] = (list, set, tuple),
) -> Generator:
    """
    Utility function to jointly flatten two arbitrality nested combination of
    iterables, and ensure the type and len of the elements of both inputs is
    similar along the way.

    Parameters
    ----------
    nested_x: Sequence
        A possibly nested sequence to flatten and compare with nested_y
    nested_y: Sequence
        A possibly nested sequence to flatten and compare with nested_x
    unpack_type: Tuple[Type], default = (list, set, tuple)
        The dobject types considered for flattening. Can be replaced by
        collections.abc.Iterable for the broadest unpacking policy, but
        this is not tested.

    Note : The function only unpacks the types provided as part of the
    `unpack_type` argument."""

    for idx, el_x in enumerate(nested_x):
        el_y = nested_y[idx]
        if isinstance(el_x, unpack_types):
            assert isinstance(el_y, type(el_x))
            assert len(el_x) == len(el_y)
            yield from flatten_and_assert(el_x, el_y)
        else:
            yield el_x, el_y


def assert_batch_equal(
    result: Sequence, expected: Sequence, framework: str
) -> None:
    """Utility function to test that a batch of the declearn.typing.Batch
    type is equal to an expected, numpy-based declearn.typing.Batch output.
    """
    # Flatten and assert type and shape of the arbitrarily nested batch
    gen = flatten_and_assert(result, expected)
    # Check all elements are equal
    for out in gen:
        res, exp = out
        # batchj element is None
        if res is None:
            assert exp is None
        # batch element is a tensor
        else:
            res = to_numpy(res, framework)
            assert_array_equal(res, exp)
