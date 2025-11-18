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

"""Unit tests for 'declearn.utils._register' tools."""

import time

import pytest

from declearn.utils import (
    access_registered,
    access_registration_info,
    access_types_mapping,
    create_types_registry,
    register_type,
)


def test_create_types_registry() -> None:
    """Unit tests for 'create_types_registry'."""
    group = f"test_{time.time_ns()}"

    class AnyClass:  # pylint: disable=all
        pass

    assert create_types_registry(AnyClass, group) is AnyClass
    with pytest.raises(KeyError):
        create_types_registry(AnyClass, group)


def test_register_type() -> None:
    """Unit tests for 'register_type' using valid instructions."""

    # Define mock custom classes.
    class BaseClass:  # pylint: disable=all
        pass

    class ChildClass(BaseClass):  # pylint: disable=all
        pass

    # Create a registry and register BaseClass.
    group = f"test_{time.time_ns()}"
    create_types_registry(BaseClass, group)
    assert register_type(BaseClass, name="base", group=group) is BaseClass
    # Register ChildClass.
    assert register_type(ChildClass, name="child", group=group) is ChildClass

    # Register another BaseClass-inheriting class using decorator syntax.
    @register_type(name="other", group=group)
    class OtherChild(BaseClass):
        pass


def test_register_type_fails() -> None:
    """Unit tests for 'register_type' using invalid instructions."""

    # Define mock custom classes.
    class BaseClass:  # pylint: disable=all
        pass

    class OtherClass:  # pylint: disable=all
        pass

    # Try registering in a group that does not exist.
    group = f"test_{time.time_ns()}"
    with pytest.raises(KeyError):
        register_type(BaseClass, name="base", group=group)
    # Try registering in any group, with no valid parent group.
    with pytest.raises(TypeError):
        register_type(BaseClass, name="base", group=None)
    # Try registering in a group with wrong class constraints.
    create_types_registry(BaseClass, group)
    with pytest.raises(TypeError):
        register_type(OtherClass, name="other", group=group)
    # Try registering the same name twice.
    register_type(BaseClass, name="base", group=group)
    with pytest.raises(KeyError):
        register_type(BaseClass, name="base", group=group)


def test_access_registered() -> None:
    """Unit tests for 'access_registered'."""

    # Define a mock custom class.
    class Class:  # pylint: disable=all
        pass

    # Register the class.
    name = f"test_{time.time_ns()}"
    create_types_registry(Class, name)
    register_type(Class, name=name, group=name)
    # Test that it can be recovered, even without specifying the group name.
    assert access_registered(name, group=name) is Class
    assert access_registered(name, group=None) is Class
    # Test that invalid instructions fail.
    name_2 = f"test_{time.time_ns()}"
    with pytest.raises(KeyError):
        access_registered(name_2, group=name)  # invalid name under group
    with pytest.raises(KeyError):
        access_registered(name_2, group=None)  # invalid name under any group
    with pytest.raises(KeyError):
        access_registered(name, group=name_2)  # non-existing group


def test_register_unspecified_group() -> None:
    """Unit tests for type-registration with implicit group membership."""
    group = f"test_{time.time_ns()}"

    # Define a parent class and an associted type registry.
    @create_types_registry(name=group)
    class Parent:  # pylint: disable=all
        pass

    # Define a child class, and register it without specifying the group.
    @register_type(name="new-child")
    class Child(Parent):  # pylint: disable=all
        pass

    # Verify that the class was put into the proper group.
    assert access_registered("new-child") is Child
    assert access_registration_info(Child) == ("new-child", group)


def test_access_registeration_info() -> None:
    """Unit tests for 'access_registration_info'."""

    # Define a pair of mock custom class.
    class Class_1:  # pylint: disable=all
        pass

    class Class_2:  # pylint: disable=all
        pass

    # Register the first class but not the second.
    name = f"test_{time.time_ns()}"
    create_types_registry(Class_1, name)
    register_type(Class_1, name=name, group=name)
    # Test that its registration info are properly recovered.
    assert access_registration_info(Class_1, group=name) == (name, name)
    assert access_registration_info(Class_1, group=None) == (name, name)
    # Test that invalid instructions fail.
    with pytest.raises(KeyError):
        access_registration_info(Class_1, group=f"test_{time.time_ns()}")
    with pytest.raises(KeyError):
        access_registration_info(Class_2, group=name)
    with pytest.raises(KeyError):
        access_registration_info(Class_2, group=None)


def test_access_types_mapping() -> None:
    """Unit tests for 'access_types_mapping'."""
    group = f"test_{time.time_ns()}"

    # Define mock custom type-registered classes.
    @register_type(name="base", group=group)
    @create_types_registry(name=group)
    class BaseClass:  # pylint: disable=all
        pass

    @register_type(name="child", group=group)
    class ChildClass(BaseClass):  # pylint: disable=all
        pass

    # Test that the created mapping may be accessed.
    mapping = access_types_mapping(group=group)
    assert mapping == {"base": BaseClass, "child": ChildClass}

    # Test that the accessed mapping is a copy, with no side effect on the
    # true underlying mapping (editable through registration functions).
    mapping["renamed"] = mapping.pop("child")
    assert mapping != access_types_mapping(group=group)
    with pytest.raises(KeyError):
        access_registered("renamed", group=group)

    # Test that the expected exception is raised for non-existing groups.
    with pytest.raises(KeyError):
        access_types_mapping(group=f"test_{time.time_ns()}")
