# coding: utf-8

"""Unit tests for 'declearn2.utils._register' tools."""

import time

import pytest

from declearn2.utils import (
    access_registered,
    access_registration_info,
    create_types_registry,
    register_type
)


def test_create_types_registry() -> None:
    """Unit tests for 'create_types_registry'."""
    group = f"test_{time.time_ns()}"
    assert create_types_registry(group, base=object) is None  # type: ignore
    with pytest.raises(KeyError):
        create_types_registry(group, base=object)


def test_register_type() -> None:
    """Unit tests for 'register_type' using valid instructions."""
    # Define mock custom classes.
    class BaseClass:  # pylint: disable=all
        pass

    class ChildClass(BaseClass):  # pylint: disable=all
        pass
    # Create a registry and register BaseClass.
    group = f"test_{time.time_ns()}"
    create_types_registry(group, base=BaseClass)
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
    # Try registering in a group with wrong class constraints.
    create_types_registry(group, base=BaseClass)
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
    create_types_registry(name, base=Class)
    register_type(Class, name=name, group=name)
    # Test that it can be recovered, even without specifying the group name.
    assert access_registered(name, group=name) is Class
    assert access_registered(name, group=None) is Class
    # Test that invalid instructions fail.
    name_2 = f"test_{time.time_ns()}"
    with pytest.raises(KeyError):
        access_registered(name_2, group=name)  # invalid name under group
    with pytest.raises(KeyError):
        access_registered(name, group=name_2)  # non-existing group


def test_access_registeration_info() -> None:
    """Unit tests for 'access_registration_info'."""
    # Define a pair of mock custom class.
    class Class_1:  # pylint: disable=all
        pass
    class Class_2:  # pylint: disable=all
        pass
    # Register the first class but not the second.
    name = f"test_{time.time_ns()}"
    create_types_registry(name, base=Class_1)
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
