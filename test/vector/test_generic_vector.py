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

"""Unit tests on 'declearn.model.api.Vector', using an ad-hoc subclass."""

import uuid
from typing import Any, List, Self, Tuple, Union

import numpy as np
import pytest

from declearn.model.api import Vector, VectorSpec, register_vector_type


class Scalar(float):
    """Float subclass, merely used to avoid side effects from tests."""


@register_vector_type(Scalar)
class ScalarFloatVector(Vector):  # noqa : PLW1641
    """Mock Vector subclasses operating on scalar float values."""

    def __eq__(
        self,
        other: Any,
    ) -> bool:
        return (
            isinstance(other, ScalarFloatVector)
            and self.coefs.keys() == other.coefs.keys()
            and all(val == other.coefs[key] for key, val in self.coefs.items())
        )

    def sign(
        self,
    ) -> Self:
        coefs = {
            key: Scalar((val > 0) - (val < 0))
            for key, val in self.coefs.items()
        }
        return type(self)(coefs)

    def minimum(
        self,
        other: Union[Self, float],
    ) -> Self:
        return self._apply_operation(other, min)

    def maximum(
        self,
        other: Union[Self, float],
    ) -> Self:
        return self._apply_operation(other, max)

    def sum(
        self,
    ) -> Self:
        return self

    def flatten(
        self,
    ) -> Tuple[List[float], VectorSpec]:
        raise NotImplementedError("This method is unimplemented.")

    @classmethod
    def unflatten(
        cls,
        values: List[float],
        v_spec: VectorSpec,
    ) -> Self:
        raise NotImplementedError("This method is unimplemented.")


class TestScalarFloatVector:
    """Test that the mock 'ScalarFloatVector' container works properly.

    This is done both to sanitize the remainder of tests in this file
    and to run primary tests on the underlying 'Vector' implementation.
    """

    def test_add(self):
        """Test that adding two 'ScalarFloatVector' objects works."""
        vec_a = ScalarFloatVector({"a": 0.0, "b": 1.0})
        vec_b = ScalarFloatVector({"a": 1.0, "b": 2.0})
        expect = ScalarFloatVector({"a": 1.0, "b": 3.0})
        result = vec_a + vec_b
        assert expect == result

    def test_sub(self):
        """Test that subtracting two 'ScalarFloatVector' objects works."""
        vec_a = ScalarFloatVector({"a": 0.0, "b": 1.0})
        vec_b = ScalarFloatVector({"a": 1.0, "b": 2.0})
        expect = ScalarFloatVector({"a": 1.0, "b": 1.0})
        result = vec_b - vec_a
        assert expect == result

    def test_mul(self):
        """Test that multiplying two 'ScalarFloatVector' objects works."""
        vec_a = ScalarFloatVector({"a": 0.0, "b": 1.0})
        vec_b = ScalarFloatVector({"a": 1.0, "b": 2.0})
        expect = ScalarFloatVector({"a": 0.0, "b": 2.0})
        result = vec_a * vec_b
        assert expect == result

    def test_truediv(self):
        """Test that dividing two 'ScalarFloatVector' objects works."""
        vec_a = ScalarFloatVector({"a": 2.0, "b": 1.0})
        vec_b = ScalarFloatVector({"a": 2.0, "b": 2.0})
        expect = ScalarFloatVector({"a": 1.0, "b": 0.5})
        result = vec_a / vec_b
        assert expect == result

    def test_pow(self):
        """Test that 'ScalarFloatVector ** scalar' works."""
        vector = ScalarFloatVector({"a": 2.0, "b": 1.0})
        expect = ScalarFloatVector({"a": 4.0, "b": 1.0})
        result = vector**2
        assert expect == result


class TestVectorErrors:
    """Unit tests on error-raising uses of 'declearn.model.api.Vector'."""

    def test_build_empty_coefs(self) -> None:
        """Try calling 'Vector.build({})'."""
        with pytest.raises(TypeError):
            Vector.build({})

    def test_build_unregistered_type(self) -> None:
        """Try calling 'Vector.build' on a dict of str values."""
        with pytest.raises(TypeError):
            Vector.build({"0": "test_string", "1": "test_string"})

    def test_build_multiple_coef_types(self) -> None:
        """Try calling 'Vector.build' on a mix of scalars and arrays."""
        coefs = {
            "float": Scalar(1.0),
            "numpy": np.arange(10),
        }
        with pytest.raises(TypeError):
            Vector.build(coefs)

    def test_shapes_undefined(self) -> None:
        """Test that 'ScalarFloatVector.shapes' raises NotImplementedError."""
        vector = ScalarFloatVector({"a": 0.0, "b": 1.0})
        with pytest.raises(NotImplementedError):
            vector.shapes()

    def test_dtypes_undefined(self) -> None:
        """Test that 'ScalarFloatVector.dtypes' raises NotImplementedError."""
        vector = ScalarFloatVector({"a": 0.0, "b": 1.0})
        with pytest.raises(NotImplementedError):
            vector.dtypes()

    def test_sum_vectors_with_different_keys(self) -> None:
        """Test that summing different-keys vectors raises a KeyError."""
        vec_a = ScalarFloatVector({"a": 0.0, "b": 1.0})
        vec_b = ScalarFloatVector({"a": 0.0, "c": 1.0})
        with pytest.raises(KeyError):
            vec_a + vec_b  # pylint: disable=pointless-statement

    def test_sum_vector_with_incompatible_type(self) -> None:
        """Test that summing a Vector and some incompatible object fails."""
        vec_a = ScalarFloatVector({"a": 0.0, "b": 1.0})
        other = [0.0, 1.0]
        with pytest.raises(TypeError):
            vec_a + other  # pylint: disable=pointless-statement

    def test_build_from_specs_wrong_specs_type(self) -> None:
        """Test that `Vector.build_from_specs` raises on mistyped specs."""
        with pytest.raises(TypeError):
            Vector.build_from_specs(
                [0.0, 1.0],
                v_spec="wrong-type",  # type: ignore
            )

    def test_build_from_specs_missing_vector_type(self) -> None:
        """Test that `Vector.build_from_specs` raises on unregistered type."""
        specs = VectorSpec(
            names=["a", "b"],
            shapes={"a": (1,), "b": (1,)},
            dtypes={"a": "float", "b": "float"},
            v_type=None,
        )
        with pytest.raises(KeyError):
            Vector.build_from_specs([0.0, 1.0], specs)

    def test_build_from_specs_unregistered_vector_type(self) -> None:
        """Test that `Vector.build_from_specs` raises on unregistered type."""
        specs = VectorSpec(
            names=["a", "b"],
            shapes={"a": (1,), "b": (1,)},
            dtypes={"a": "float", "b": "float"},
            v_type=(str(uuid.uuid4()), "Vector"),  # unregistered
        )
        with pytest.raises(KeyError):
            Vector.build_from_specs([0.0, 1.0], specs)

    def test_build_from_specs_wrong_vector_type(self) -> None:
        """Test that `Vector.build_from_specs` raises on non-Vector type."""
        specs = VectorSpec(
            names=["a", "b"],
            shapes={"a": (1,), "b": (1,)},
            dtypes={"a": "float", "b": "float"},
            v_type=("SklearnSGDModel", "Model"),  # not a Vector subclass
        )
        with pytest.raises(TypeError):
            Vector.build_from_specs([0.0, 1.0], specs)
