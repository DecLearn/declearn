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

"""Unit tests for `declearn.utils._serialize` tools.

Note: some of these tests require `declearn.utils._register` tools
      (which are tested in a separate script) to work as expected.
"""

import os
import tempfile
import time
from typing import Any, Dict, Optional, Tuple, Type

import pytest

from declearn.utils import (
    ObjectConfig,
    create_types_registry,
    deserialize_object,
    register_type,
    serialize_object,
)


class MockClass:
    """Mock class implementing get/from config used for testing purposes."""

    def __init__(self, val: int = 42) -> None:
        self.val = val

    def get_config(self) -> Dict[str, Any]:
        """Return the object's configuration dict."""
        return {"val": self.val}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MockClass":
        """Instantiate from a configuration dict."""
        return cls(**config)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and (self.val == other.val)


@pytest.fixture(name="registered_class")
def fixture_registered_class() -> Tuple[Type[MockClass], str]:
    """Provide with a type-registered MockClass subclass."""

    # Declare a subtype to avoid side effects between tests.
    class SubClass(MockClass):  # pylint: disable=all
        pass

    # Create a test-specific types registry and add the type to it.
    group = str(time.time_ns())
    create_types_registry(MockClass, group)
    register_type(SubClass, name="mock", group=group)
    # Return both the class constructor and its registration group.
    return SubClass, group


def test_object_config() -> None:
    """Unit tests for `ObjectConfig`."""
    # Instantiate a mock ObjectConfig instance and its expected dict form.
    config = ObjectConfig(
        name="lorem", group="ipsum", config={"a": 0, "b": [1, 2]}
    )
    c_dict = {
        "name": "lorem",
        "group": "ipsum",
        "config": {"a": 0, "b": [1, 2]},
    }
    # Test that conversion to and from dict works.
    assert ObjectConfig(**c_dict) == config  # type: ignore
    assert config.to_dict() == c_dict
    # Test that conversion to and from JSON works.
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "config.json")
        assert not os.path.isfile(path)
        config.to_json(path)
        assert os.path.isfile(path)
        assert config.from_json(path) == config


def test_serialize_unregistered() -> None:
    """Unit tests for `serialize_object` with an un-registered type."""
    obj = MockClass()
    with pytest.raises(KeyError):
        serialize_object(obj)
    cfg = serialize_object(obj, allow_unregistered=True)
    assert isinstance(cfg, ObjectConfig)
    assert cfg.name == "MockClass"
    assert cfg.group is None
    assert cfg.config == obj.get_config()


def test_serialize_registered(
    registered_class: Tuple[Type[MockClass], str]
) -> None:
    """Unit tests for `serialize_object` with a registered type."""
    cls, group = registered_class
    obj = cls()
    # This should fail due to unproper group specification.
    with pytest.raises(KeyError):
        serialize_object(obj, group=str(time.time_ns()))
    # This should work, whether the group is specified or not.
    for gkey in (group, None):
        cfg = serialize_object(obj, group=gkey)
        assert isinstance(cfg, ObjectConfig)
        assert cfg.name == "mock"
        assert cfg.group == group
        assert cfg.config == obj.get_config()


def _setup_config_inputs(
    obj: MockClass, group: Optional[str], folder: str
) -> Tuple[Dict[str, Any], ObjectConfig, str]:
    """Create three alternative input formats to `deserialize_object`."""
    cfg_dict = {"name": "mock", "group": group, "config": obj.get_config()}
    cfg_objc = ObjectConfig(**cfg_dict)  # type: ignore
    cfg_path = os.path.join(folder, "config.json")
    cfg_objc.to_json(cfg_path)
    return cfg_dict, cfg_objc, cfg_path


@pytest.mark.parametrize(
    "index", [0, 1, 2], ids=["dict", "ObjectConfig", "JSON path"]
)
def test_deserialize_unregistered(index: int) -> None:
    """Unit tests from `deserialize_object` with an unregistered type."""
    # Test that unproper inputs raise a TypeError.
    with pytest.raises(TypeError):
        deserialize_object({"lorem": "ipsum"})  # type: ignore
    # Set up a mock instance and a 'custom' type-mapping dict.
    obj = MockClass()
    group = str(time.time_ns())  # avoid other tests' side effects
    custom = {"mock": MockClass}
    with tempfile.TemporaryDirectory() as folder:
        config = _setup_config_inputs(obj, group, folder)[index]
        with pytest.raises(KeyError):
            deserialize_object(config)  # type: ignore
        assert deserialize_object(config, custom) == obj  # type: ignore


@pytest.mark.parametrize(
    "index", [0, 1, 2], ids=["dict", "ObjectConfig", "JSON path"]
)
def test_deserialize_registered(
    registered_class: Tuple[Type[MockClass], str], index: int
) -> None:
    """Unit tests from `deserialize_object` with a registered type."""
    cls, group = registered_class
    obj = cls()
    with tempfile.TemporaryDirectory() as folder:
        config = _setup_config_inputs(obj, group, folder)[index]
        assert deserialize_object(config) == obj  # type: ignore
