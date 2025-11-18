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

"""Shared unit test suite for Vector subclasses."""

import operator
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Type, TypeVar

import numpy as np

from declearn.model.api import Vector, VectorSpec
from declearn.test_utils import assert_json_serializable_dict, to_numpy

__all__ = [
    "VectorFactory",
    "VectorTestSuite",
]


VT = TypeVar("VT", bound=Vector)


class VectorFactory(Generic[VT], metaclass=ABCMeta):
    """ABC to make up Vector (subclasses) factories for unit tests."""

    # Class attributes defining the Vector type used and its framework name.
    framework: ClassVar[str]
    vector_cls: Type[VT]

    # Hard-coded specs of generated vectors.
    names = ["2-dim", "1-dim", "3-dim"]
    shapes = [(4, 2), (7,), (3, 5, 9)]
    dtypes = ["float32", "float32", "float64"]

    def make_values(
        self,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Build numpy arrays with determinisic specs and seeded-RNG values."""
        rng = np.random.default_rng(seed)
        return {
            name: (
                rng.normal(size=shape).astype(dtype)
                if dtype.startswith("float")
                else rng.uniform(1, 10, size=shape).astype(dtype)
            )
            for name, shape, dtype in zip(
                self.names, self.shapes, self.dtypes, strict=False
            )
        }

    @abstractmethod
    def make_vector(
        self,
        seed: int = 0,
    ) -> VT:
        """Build a Vector of deterministic specs, with seeded-RNG values."""

    def assert_equal(
        self,
        expect: Dict[str, np.ndarray],
        vector: VT,
    ) -> None:
        """Raise an AssertionError if a Vector mismatches expected values."""
        assert all(
            np.allclose(expect[key], to_numpy(val, self.framework))
            for key, val in vector.coefs.items()
        )


class VectorSelfOpTests:
    """Unit tests for Vector self-based operations."""

    def test_shapes(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that shapes can properly be accessed and abide by specs."""
        vector = factory.make_vector(seed=0)
        shapes = vector.shapes()
        assert shapes.keys() == vector.coefs.keys()
        assert all(
            isinstance(shape, tuple) and all(isinstance(d, int) for d in shape)
            for shape in shapes.values()
        )

    def test_dtypes(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that dtypes can properly be accessed and abide by specs."""
        vector = factory.make_vector(seed=0)
        dtypes = vector.dtypes()
        assert dtypes.keys() == vector.coefs.keys()
        assert all(isinstance(dtype, str) for dtype in dtypes.values())

    def test_repr(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that a Vector's repr contains expected information."""
        vector = factory.make_vector(seed=0)
        result = repr(vector)
        assert isinstance(result, str)
        assert type(vector).__name__ in result
        assert all(name in result for name in factory.names)
        assert all(dtype in result for dtype in factory.dtypes)
        assert all(str(shape) in result for shape in factory.shapes)

    def test_pack_unpack(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that a Vector can be (de)serialized to, then from, a dict."""
        vector = factory.make_vector(seed=0)
        data = vector.pack()
        assert_json_serializable_dict(data)
        vecbis = type(vector).unpack(data)
        assert vector == vecbis

    def test_build(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that a Vector can be re-created via the `Vector.build` generic.

        This indirectly verifies the proper type-registration of the
        tested Vector subclass and of its associate data types.
        """
        vector = factory.make_vector(seed=0)
        vecbis = Vector.build(vector.coefs)
        assert isinstance(vecbis, type(vector))
        assert vector == vecbis

    def test_flatten(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that a Vector's `flatten` method outputs proper-type data."""
        vector = factory.make_vector(seed=0)
        values, v_spec = vector.flatten()
        assert isinstance(values, list)
        assert all(isinstance(x, float) for x in values)
        assert isinstance(v_spec, VectorSpec)
        assert v_spec.names == factory.names
        assert v_spec.shapes == dict(
            zip(factory.names, factory.shapes, strict=False)
        )
        assert v_spec.dtypes == dict(
            zip(factory.names, factory.dtypes, strict=False)
        )
        assert isinstance(v_spec.v_type, tuple)
        assert len(v_spec.v_type) == 2
        assert all(isinstance(s, str) for s in v_spec.v_type)

    def test_flatten_unflatten(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that a Vector can be flattened and then unflattened."""
        vector = factory.make_vector(seed=0)
        values, v_spec = vector.flatten()
        vecbis = type(vector).unflatten(values, v_spec)
        assert vector == vecbis

    def test_build_from_specs(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that the `Vector.build_from_specs` generic works.

        It is designed to enable unflattening a Vector from its specs.
        """
        vector = factory.make_vector(seed=0)
        values, v_spec = vector.flatten()
        vecbis = Vector.build_from_specs(values, v_spec)
        assert isinstance(vecbis, type(vector))
        assert vector == vecbis

    def test_sign(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that the sign operator of a Vector works properly."""
        vector = factory.make_vector(seed=0)
        expect = {
            key: np.sign(to_numpy(val, factory.framework))
            for key, val in vector.coefs.items()
        }
        result = vector.sign()
        factory.assert_equal(expect, result)

    def test_sum(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that the sum-reduce operator of a Vector works properly."""
        vector = factory.make_vector(seed=0)
        expect: Dict[str, np.ndarray] = {
            key: np.sum(to_numpy(val, factory.framework))
            for key, val in vector.coefs.items()
        }
        result = vector.sum()
        factory.assert_equal(expect, result)


class VectorVectorOpTests:
    """Unit tests for operating on two same-class, same-specs Vector."""

    def _test_op_vector(
        self,
        factory: VectorFactory,
        tested_op: Callable[[Any, Any], Any],
    ) -> None:
        """Backend for vector-with-vector operation unit tests."""
        vec_a = factory.make_vector(seed=0)
        vec_b = factory.make_vector(seed=1)
        expect = {
            key: tested_op(
                to_numpy(vec_a.coefs[key], factory.framework),
                to_numpy(vec_b.coefs[key], factory.framework),
            )
            for key in vec_a.coefs
        }
        result = tested_op(vec_a, vec_b)
        factory.assert_equal(expect, result)

    def test_add_vector(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that the addition of two same-spec Vector works properly."""
        self._test_op_vector(factory, operator.add)

    def test_sub_vector(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that the subtraction of two same-spec Vector works properly."""
        self._test_op_vector(factory, operator.sub)

    def test_mul_vector(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that multiplying two same-spec Vector works properly."""
        self._test_op_vector(factory, operator.mul)

    def test_truediv_vector(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that the division of two same-spec Vector works properly."""
        self._test_op_vector(factory, operator.truediv)

    def test_minimum_vector(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test computing the element-wise minimum of two same-spec Vector."""
        vec_a = factory.make_vector(seed=0)
        vec_b = factory.make_vector(seed=1)
        expect = {
            key: np.minimum(
                to_numpy(vec_a.coefs[key], factory.framework),
                to_numpy(vec_b.coefs[key], factory.framework),
            )
            for key in vec_a.coefs
        }
        result = vec_a.minimum(vec_b)
        factory.assert_equal(expect, result)

    def test_maximum_vector(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test computing the element-wise maximum of two same-spec Vector."""
        vec_a = factory.make_vector(seed=0)
        vec_b = factory.make_vector(seed=1)
        expect = {
            key: np.maximum(
                to_numpy(vec_a.coefs[key], factory.framework),
                to_numpy(vec_b.coefs[key], factory.framework),
            )
            for key in vec_a.coefs
        }
        result = vec_a.maximum(vec_b)
        factory.assert_equal(expect, result)

    def test_eq_same_vectors(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test the equality operator on two identical Vector instances."""
        vec_a = factory.make_vector(seed=0)
        vec_b = factory.make_vector(seed=0)
        assert vec_a == vec_b

    def test_eq_same_specs(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test the (in)equality operator on same-spec, diff. values Vector."""
        vec_a = factory.make_vector(seed=0)
        vec_b = factory.make_vector(seed=1)
        assert vec_a != vec_b

    def test_eq_different_specs(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test the (in)equality operator on two different-spec Vector."""
        vector = factory.make_vector(seed=0)
        vecbis = type(vector)(
            {key: vector.coefs[key] for key in list(vector.coefs)[:1]}
        )
        assert vector != vecbis


class VectorScalarOpTests:
    """Unit tests for operating a Vector with a scalar value."""

    @staticmethod
    def _generate_scalar() -> float:
        """Generate a random non-zero float value."""
        while (scalar := np.random.normal()) == 0.0:
            continue
        return float(scalar)

    def _test_op_scalar_left(
        self,
        factory: VectorFactory,
        scalar: float,
        tested_op: Callable[[Any, Any], Any],
    ) -> None:
        """Test that a vector <op> scalar operation works properly."""
        vector = factory.make_vector(seed=0)
        expect = {
            key: tested_op(to_numpy(val, factory.framework), scalar)
            for key, val in vector.coefs.items()
        }
        result = tested_op(vector, scalar)
        factory.assert_equal(expect, result)

    def _test_op_scalar_right(
        self,
        factory: VectorFactory,
        scalar: float,
        tested_op: Callable[[Any, Any], Any],
    ) -> None:
        """Test that a scalar <op> vector operation works properly."""
        vector = factory.make_vector(seed=0)
        expect = {
            key: tested_op(scalar, to_numpy(val, factory.framework))
            for key, val in vector.coefs.items()
        }
        result = tested_op(scalar, vector)
        factory.assert_equal(expect, result)

    def test_add_scalar(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that `vector + scalar` works properly."""
        scalar = self._generate_scalar()
        self._test_op_scalar_left(factory, scalar, operator.add)

    def test_radd_scalar(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that `scalar + vector` works properly."""
        scalar = self._generate_scalar()
        self._test_op_scalar_right(factory, scalar, operator.add)

    def test_sub_scalar(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that `vector - scalar` works properly."""
        scalar = self._generate_scalar()
        self._test_op_scalar_left(factory, scalar, operator.sub)

    def test_rsub_scalar(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that `scalar - vector` works properly."""
        scalar = self._generate_scalar()
        self._test_op_scalar_right(factory, scalar, operator.sub)

    def test_mul_scalar(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that `vector * scalar` works properly."""
        scalar = self._generate_scalar()
        self._test_op_scalar_left(factory, scalar, operator.mul)

    def test_rmul_scalar(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that `scalar * vector` works properly."""
        scalar = self._generate_scalar()
        self._test_op_scalar_right(factory, scalar, operator.mul)

    def test_truediv_scalar(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that `vector / scalar` works properly."""
        scalar = self._generate_scalar()
        self._test_op_scalar_left(factory, scalar, operator.truediv)

    def test_rtruediv_scalar(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that `scalar / vector` works properly."""
        scalar = self._generate_scalar()
        self._test_op_scalar_right(factory, scalar, operator.truediv)

    def test_pow_scalar(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that `vector ** scalar_positive_int` works properly."""
        scalar = round(abs(self._generate_scalar() * 10))
        self._test_op_scalar_left(factory, scalar, operator.pow)

    def test_square(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that square operation (via power) works."""
        vector = factory.make_vector(seed=0)
        expect = {
            key: np.square(to_numpy(val, factory.framework))
            for key, val in vector.coefs.items()
        }
        result = vector**2
        factory.assert_equal(expect, result)

    def test_square_root(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test that square root operation (via power) works."""
        vector = factory.make_vector(seed=0)
        vector = vector * vector.sign()  # positive values only
        expect = {
            key: np.sqrt(to_numpy(val, factory.framework))
            for key, val in vector.coefs.items()
        }
        result = vector**0.5
        factory.assert_equal(expect, result)

    def test_minimum_scalar(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test element-wise minimum of a Vector with a scalar."""
        vector = factory.make_vector(seed=0)
        scalar = self._generate_scalar()
        expect = {
            key: np.minimum(val, scalar)
            for key, val in factory.make_values(seed=0).items()
        }
        result = vector.minimum(scalar)
        factory.assert_equal(expect, result)

    def test_maximum_scalar(
        self,
        factory: VectorFactory,
    ) -> None:
        """Test element-wise maximum of a Vector with a scalar."""
        vector = factory.make_vector(seed=0)
        scalar = self._generate_scalar()
        expect = {
            key: np.maximum(val, scalar)
            for key, val in factory.make_values(seed=0).items()
        }
        result = vector.maximum(scalar)
        factory.assert_equal(expect, result)


class VectorTestSuite(
    VectorSelfOpTests,
    VectorVectorOpTests,
    VectorScalarOpTests,
):
    """Shared unit test suite for all Vector subclasses."""
