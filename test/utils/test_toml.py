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

"""Unit tests for the `TomlConfig` util."""

import dataclasses
import os
import warnings
from typing import Any, Dict, List, Optional, Self, Tuple, Union

import pytest

from declearn.utils import TomlConfig


class Custom:  # noqa: PLW1641
    """Custom class that requires specific TOML parsing."""

    def __init__(
        self,
        value: Any = "default",
        _from_config: bool = False,
    ) -> None:
        """Default builder for Custom instances."""
        self.value = value
        self.f_cfg = _from_config

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Custom):
            return self.value == other.value
        return NotImplemented

    @classmethod
    def from_config(cls, **kwargs: Any) -> Self:
        """Alternative builder for Custom instances."""
        return cls(_from_config=True, **kwargs)


@dataclasses.dataclass
class DemoTomlConfig(TomlConfig):
    """Demonstration TomlConfig subclass."""

    req_int: int
    req_lst: List[str]
    opt_str: str = "default"
    opt_tup: Optional[Tuple[int, int]] = None
    opt_dct: Dict[str, float] = dataclasses.field(default_factory=dict)
    opt_obj: Custom = dataclasses.field(default_factory=Custom)
    opt_unc: Union[str, Custom, None] = None

    @classmethod
    def parse_opt_tup(
        cls,
        field: dataclasses.Field,
        inputs: Any,
    ) -> Optional[Tuple[int, int]]:
        """Custom parser for `opt_tup`, adding list-to-tuple conversion."""
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        return cls.default_parser(field, inputs)

    @classmethod
    def get_field(cls, name: str) -> dataclasses.Field:
        """Access the definition of a given dataclass field."""
        return {field.name: field for field in dataclasses.fields(cls)}[name]


class TestTomlConfigDefaultParser:
    """Unit tests for `TomlConfig.default_parser`, using a demo subclass."""

    def test_int(self) -> None:
        """Test that the parser works for an int field."""
        field = DemoTomlConfig.get_field("req_int")
        assert TomlConfig.default_parser(field, 42) == 42
        with pytest.raises(TypeError):
            TomlConfig.default_parser(field, 42.0)

    def test_lst(self) -> None:
        """Test that the parser works for a list of str field."""
        field = DemoTomlConfig.get_field("req_lst")
        # Test with a list of str.
        value = ["this", "is", "a", "test"]
        assert TomlConfig.default_parser(field, value) is value
        # Test with an empty list.
        value = []
        assert TomlConfig.default_parser(field, value) is value
        # Test with a single str.
        with pytest.raises(TypeError):
            TomlConfig.default_parser(field, None)
        # Test with a mixed-type list.
        with pytest.raises(TypeError):
            TomlConfig.default_parser(field, ["this", "fails", 0])

    def test_opt_str(self) -> None:
        """Test that the parser works for an optional str field."""
        field = DemoTomlConfig.get_field("opt_str")
        # Test without a value.
        assert TomlConfig.default_parser(field, None) is field.default
        # Test with a valid value.
        assert TomlConfig.default_parser(field, "test") == "test"
        # Test with an invalid value.
        with pytest.raises(TypeError):
            TomlConfig.default_parser(field, 0)

    def test_opt_tup(self) -> None:
        """Test that the parser works for an optional tuple of int field."""
        field = DemoTomlConfig.get_field("opt_tup")
        # Test without a value.
        assert TomlConfig.default_parser(field, None) is field.default
        # Test with a valid value.
        value = (12, 15)
        assert TomlConfig.default_parser(field, value) is value
        # Test with invalid values (wrong type, length or internal types).
        with pytest.raises(TypeError):
            TomlConfig.default_parser(field, [12, 15])
        with pytest.raises(TypeError):
            TomlConfig.default_parser(field, (12, 15, 18))
        with pytest.raises(TypeError):
            TomlConfig.default_parser(field, (12, "15"))

    def test_opt_dct(self) -> None:
        """Test that the parser works for an optional dict field."""
        field = DemoTomlConfig.get_field("opt_dct")
        # Test without a value.
        assert TomlConfig.default_parser(field, None) == {}
        # Test with a valid value.
        value = {"a": 0.0, "b": 1.0}
        assert TomlConfig.default_parser(field, value) is value
        # Test with invalid values (wrong type, key types or value types).
        with pytest.raises(TypeError):
            TomlConfig.default_parser(field, "test")
        with pytest.raises(TypeError):
            TomlConfig.default_parser(field, {0: 0.0})  # type: ignore
        with pytest.raises(TypeError):
            TomlConfig.default_parser(field, {"a": "val"})

    def test_opt_obj(self) -> None:
        """Test that the parser works for an optional custom object field."""
        field = DemoTomlConfig.get_field("opt_obj")
        # Test without a value.
        assert TomlConfig.default_parser(field, None) == Custom()
        # Test with a valid value.
        value = Custom(value=42.0)
        assert TomlConfig.default_parser(field, value) is value
        # Test with kwargs for the object itself.
        built = TomlConfig.default_parser(field, {"value": 0.0})
        assert isinstance(built, Custom)
        assert built.value == 0.0
        # Test with an invalid value.
        with pytest.raises(TypeError):
            TomlConfig.default_parser(field, "invalid")

    def test_opt_unc(self) -> None:
        """Test that the parser works for a union with custom types."""
        field = DemoTomlConfig.get_field("opt_unc")
        # Test without a value.
        assert TomlConfig.default_parser(field, None) is None
        # Test with a valid str value.
        assert TomlConfig.default_parser(field, "unc") == "unc"
        # Test with a valid Custom object.class TestTomlConfig:
        value = Custom(value="test")
        assert TomlConfig.default_parser(field, value) is value
        # Test with kwargs for a Custom object.
        built = TomlConfig.default_parser(field, {"value": "test"})
        assert isinstance(built, Custom)
        assert built.value == "test"
        # Test with an invalid value.
        with pytest.raises(TypeError):
            TomlConfig.default_parser(field, {"invalid": "kwarg"})


class TestTomlConfigFromParams:
    """Unit tests for `TomlConfig.from_params`, using a demo subclass."""

    exhaustive_params = {
        "req_int": 0,
        "req_lst": ["test"],
        "opt_str": "test",
        "opt_tup": (0, 1),
        "opt_dct": {"key": 0.0},
        "opt_obj": Custom(value=0),
        "opt_unc": Custom(value=1),
    }

    def test_all_params(self) -> None:
        """Test that parsing from an exhaustive dict of valid params works."""
        parsed = DemoTomlConfig.from_params(**self.exhaustive_params)
        assert isinstance(parsed, DemoTomlConfig)
        for key, val in self.exhaustive_params.items():
            assert getattr(parsed, key) == val

    def test_partial_params(self) -> None:
        """Test that parsing without some optional params works."""
        params = {"req_int": 0, "req_lst": ["test"]}
        parsed = DemoTomlConfig.from_params(**params)
        assert isinstance(parsed, DemoTomlConfig)
        assert all(getattr(parsed, key) == val for key, val in params.items())

    def test_bad_params(self) -> None:
        """Test that parsing with some bad params fails."""
        # Missing required keys.
        with pytest.raises(RuntimeError):
            DemoTomlConfig.from_params(req_int=0, opt_str="test")
        # Invalid value types.
        with pytest.raises(RuntimeError):
            DemoTomlConfig.from_params(req_int=0, req_lst=1)

    def test_extra_params(self) -> None:
        """Test that provided with extra parameters raises a warning."""
        with pytest.warns(RuntimeWarning):
            DemoTomlConfig.from_params(req_int=0, req_lst=["1"], extra=2)


class TestTomlConfigFromToml:
    """Unit tests for `TomlConfig.from_toml`, using a demo subclass."""

    exhaustive_toml = """
    req_int = 0
    req_lst = ["test"]
    opt_str = "test"
    opt_tup = [0, 1]
    opt_dct = {key = 0.0}
    opt_obj = {value = 0}
    opt_unc = {value = 1}
    """

    def test_from_exhaustive_toml(self, tmp_path: str) -> None:
        """Test that parsing from an exhaustive and valid TOML file works."""
        # Export the TOML config file.
        path = os.path.join(tmp_path, "config.toml")
        with open(path, "w", encoding="utf-8") as file:
            file.write(self.exhaustive_toml)
        # Parse it and verify the outputs.
        parsed = DemoTomlConfig.from_toml(path)
        assert isinstance(parsed, DemoTomlConfig)
        for key, val in TestTomlConfigFromParams.exhaustive_params.items():
            assert getattr(parsed, key) == val

    def test_wrong_file_fails(self, tmp_path: str) -> None:
        """Test that a proper error is raised when parsing an invalid file."""
        # Export the non-TOML file.
        path = os.path.join(tmp_path, "config.toml")
        with open(path, "w", encoding="utf-8") as file:
            file.write("this is not a TOML file")
        # Verify that the parsing error is properly caught and wrapped.
        with pytest.raises(RuntimeError):
            DemoTomlConfig.from_toml(path)

    def test_warn_user(self, tmp_path: str) -> None:
        """Test that the 'warn_user' boolean flag works properly."""
        # Export the TOML config file, with an extra field.
        path = os.path.join(tmp_path, "config.toml")
        with open(path, "w", encoding="utf-8") as file:
            file.write(self.exhaustive_toml + "\nunused = nan")
        # Parse it, expecting a warning.
        with pytest.warns(RuntimeWarning):
            DemoTomlConfig.from_toml(path)
        # Parse it again, expecting no warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            DemoTomlConfig.from_toml(path, warn_user=False)

    def test_use_section(self, tmp_path: str) -> None:
        """Test that the 'use_section' option works properly."""
        # Export the TOML config file, within a wrapping section.
        path = os.path.join(tmp_path, "config.toml")
        with open(path, "w", encoding="utf-8") as file:
            file.write("[mysection]\n" + self.exhaustive_toml)
        # Verify that a basic attempt at parsing fails.
        with pytest.raises(RuntimeError):
            DemoTomlConfig.from_toml(path)
        # Parse it and verify the outputs.
        parsed = DemoTomlConfig.from_toml(path, use_section="mysection")
        assert isinstance(parsed, DemoTomlConfig)
        for key, val in TestTomlConfigFromParams.exhaustive_params.items():
            assert getattr(parsed, key) == val

    def test_use_section_fails(self, tmp_path: str) -> None:
        """Test that the 'use_section' option fails properly."""
        # Export the TOML config file.
        path = os.path.join(tmp_path, "config.toml")
        with open(path, "w", encoding="utf-8") as file:
            file.write(self.exhaustive_toml)
        # Verify that targetting a non-existing section fails.
        with pytest.raises(KeyError):
            DemoTomlConfig.from_toml(path, use_section="mysection")
        # Verify that the error can be skipped using `section_fails_ok`.
        parsed = DemoTomlConfig.from_toml(
            path, use_section="mysection", section_fail_ok=True
        )
        assert isinstance(parsed, DemoTomlConfig)
        for key, val in TestTomlConfigFromParams.exhaustive_params.items():
            assert getattr(parsed, key) == val


@dataclasses.dataclass
class ComplexTomlConfig(TomlConfig):
    """Demonstration TomlConfig subclass with some nestedness."""

    demo_a: DemoTomlConfig
    demo_b: DemoTomlConfig


class TestTomlConfigNested:
    """Unit test for a complex, nested 'TomlConfig' subclass."""

    def test_multisection_config(self, tmp_path: str) -> None:
        """Test parsing a multi-section TOML file using nested parsers."""
        # Export the multi-section TOML config file.
        path = os.path.join(tmp_path, "config.toml")
        toml = (
            f"[demo_a]\n{TestTomlConfigFromToml.exhaustive_toml}\n"
            f"[demo_b]\n{TestTomlConfigFromToml.exhaustive_toml}\n"
        )
        with open(path, "w", encoding="utf-8") as file:
            file.write(toml)
        # Verify that is can be properly parsed.
        parsed = ComplexTomlConfig.from_toml(path)
        assert isinstance(parsed, ComplexTomlConfig)
        for fname in ("demo_a", "demo_b"):
            field = getattr(parsed, fname)
            assert isinstance(field, DemoTomlConfig)
            for key, val in TestTomlConfigFromParams.exhaustive_params.items():
                assert getattr(field, key) == val

    def test_multifiles_config(self, tmp_path: str) -> None:
        """Test parsing a multi-files TOML config using nested parsers."""
        # Export the multi-files TOML config.
        # demo_a: section file
        path_a = os.path.join(tmp_path, "demo_a.toml")
        with open(path_a, "w", encoding="utf-8") as file:
            file.write(f"[demo_a]\n{TestTomlConfigFromToml.exhaustive_toml}")
        # demo_b: full-file
        path_b = os.path.join(tmp_path, "demo_b.toml")
        with open(path_b, "w", encoding="utf-8") as file:
            file.write(TestTomlConfigFromToml.exhaustive_toml)
        # main config file
        path = os.path.join(tmp_path, "config.toml")
        with open(path, "w", encoding="utf-8") as file:
            file.write(f"demo_a = '{path_a}'\ndemo_b = '{path_b}'")
        # Verify that is can be properly parsed.
        parsed = ComplexTomlConfig.from_toml(path)
        assert isinstance(parsed, ComplexTomlConfig)
        for fname in ("demo_a", "demo_b"):
            field = getattr(parsed, fname)
            assert isinstance(field, DemoTomlConfig)
            for key, val in TestTomlConfigFromParams.exhaustive_params.items():
                assert getattr(field, key) == val

    def test_multifiles_config_fails(self, tmp_path: str) -> None:
        """Test parsing a multi-files TOML config with invalid contents."""
        # Export a badly-formatted secondary file.
        path_bad = os.path.join(tmp_path, "bad.toml")
        with open(path_bad, "w", encoding="utf-8") as file:
            file.write("this file is not a TOML one")
        # Export a main config file that points to the unproper one.
        path = os.path.join(tmp_path, "config.toml")
        with open(path, "w", encoding="utf-8") as file:
            file.write(f"demo_a = '{path_bad}'\ndemo_b = '{path_bad}'")
        # Verify the the parsing error is properly caught and wrapped.
        with pytest.raises(RuntimeError):
            ComplexTomlConfig.from_toml(path)
        field = {
            field.name: field
            for field in dataclasses.fields(ComplexTomlConfig)
        }["demo_a"]
        with pytest.raises(TypeError):
            ComplexTomlConfig.default_parser(field, path_bad)


@dataclasses.dataclass
class AutofillTomlConfig(TomlConfig):
    """Demonstration TomlConfig subclass with an autofill field."""

    base: int
    auto: int

    autofill_fields = {"auto"}

    @classmethod
    def from_params(
        cls,
        **kwargs: Any,
    ) -> Self:
        if "base" in kwargs:
            kwargs.setdefault("auto", kwargs["base"])
        return super().from_params(**kwargs)


class TestTomlAutofill:
    """Unit tests for a 'TomlConfig' subclass with an auto-fill field."""

    def test_from_params_exhaustive(self) -> None:
        """Test parsing kwargs with exhaustive values."""
        config = AutofillTomlConfig.from_params(base=0, auto=1)
        assert config.base == 0  # false-positive; pylint: disable=no-member
        assert config.auto == 1  # false-positive; pylint: disable=no-member

    def test_from_params_autofill(self) -> None:
        """Test parsing kwargs without the auto-filled value."""
        config = AutofillTomlConfig.from_params(base=0)
        assert config.base == 0  # false-positive; pylint: disable=no-member
        assert config.auto == 0  # false-positive; pylint: disable=no-member

    def test_from_toml_exhaustive(self, tmp_path: str) -> None:
        """Test parsing a TOML file with exhaustive values."""
        path = os.path.join(tmp_path, "config.toml")
        with open(path, "w", encoding="utf-8") as file:
            file.write("base = 0\nauto = 1")
        config = AutofillTomlConfig.from_toml(path)
        assert config.base == 0  # false-positive; pylint: disable=no-member
        assert config.auto == 1  # false-positive; pylint: disable=no-member

    def test_from_toml_autofill(self, tmp_path: str) -> None:
        """Test parsing a TOML file without the auto-filled value."""
        path = os.path.join(tmp_path, "config.toml")
        with open(path, "w", encoding="utf-8") as file:
            file.write("base = 0")
        config = AutofillTomlConfig.from_toml(path)
        assert config.base == 0  # false-positive; pylint: disable=no-member
        assert config.auto == 0  # false-positive; pylint: disable=no-member
