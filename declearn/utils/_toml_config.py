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

"""Base class to define TOML-parsable configuration containers."""

import dataclasses
import os
import tomllib  # type: ignore
import typing
import warnings
from typing import (
    Any,
    ClassVar,
    Dict,
    Optional,
    Self,
    Set,
    Type,
    TypeVar,
    Union,
)

__all__ = [
    "TomlConfig",
]


T = TypeVar("T")


def _isinstance_generic(inputs: Any, typevar: Type) -> bool:
    """Override of `isinstance` built-in that supports some typing generics.

    Note
    ----
    This function was mainly implemented to make up for the lack of a
    backport of `isinstance(x, Optional[T])` for Python <3.9. Thus it
    is currently private and used in a single place, with its future
    (being kept / improved / made public / removed) being unclear.

    Parameters
    ----------
    inputs: <T>
        Input instance that needs type-checking.
    typevar: type or typing generic
        Type hint based on which to type-check `inputs`.
        May be a base type, or a typing generic among a (limited)
        number of supported cases: Dict, List, Tuple, Union.
        Note that Optional is supported as it is a Union alias.

    Returns
    -------
    is_instance: bool
        Whether `inputs` abides by the type specified by `typevar`.
        For composed type generics, recursive type-checks may be
        conducted (e.g. to type-check elements of an iterable).

    Raises
    ------
    TypeError
        If an unsupported `typevar` is provided.
    """
    origin = typing.get_origin(typevar)
    # Case of a raw, unit type.
    if origin is None:
        return (typevar is Any) or isinstance(inputs, typevar)
    # Case of a typing generic.
    args = typing.get_args(typevar)
    # Case of a Union generic.
    if origin is typing.Union:
        return any(_isinstance_generic(inputs, typevar) for typevar in args)
    # Case of a Dict[..., ...] generic.
    if origin is dict:
        return (
            isinstance(inputs, dict)
            and all(_isinstance_generic(k, args[0]) for k in inputs)
            and all(_isinstance_generic(v, args[1]) for v in inputs.values())
        )
    # Case of a List[...] generic.
    if origin is list:
        return isinstance(inputs, list) and all(
            _isinstance_generic(e, args[0]) for e in inputs
        )
    # Case of a Tuple[...] generic.
    if origin is tuple:
        return (
            isinstance(inputs, tuple)
            and len(inputs) == len(args)
            and all(
                _isinstance_generic(e, t)
                for e, t in zip(inputs, args, strict=False)
            )
        )
    # Unsupported cases.
    raise TypeError(  # pragma: no cover
        "Unsupported subscripted generic for instance check: "
        f"'{typevar}' with origin '{origin}'."
    )


def _parse_float(src: str) -> Optional[float]:
    """Custom float parser that replaces nan values with None."""
    return None if src == "nan" else float(src)


def _instantiate_field(
    field: dataclasses.Field[T],
    **kwargs: Any,
) -> T:
    """Instantiate a dataclass field from input args and kwargs.

    This function is meant to enable building dataclass object fields,
    that requring instantiating from a received value or dict of kwargs.

    It supports doing so even when for fields that are annotated to be a
    union of types (notably optional ones: Union[T, None]), and/or are a
    `TomlConfig` subclass that should be instantiated via `from_params`
    rather than `cls(*args, **kwargs)`.

    It will raise a TypeError if instantiation fails or if `field.type`
    has and unsupported typing origin. It may also raise any exception
    coming from the target type's `__init__` method.
    """

    def _instantiate(cls: Type[Any]) -> Any:
        """Try instantiating a given class from the kwargs."""
        if issubclass(cls, TomlConfig):
            return cls.from_params(**kwargs)
        return cls(**kwargs)

    origin = typing.get_origin(field.type)
    # Case of a raw type.
    if origin is None:
        return _instantiate(field.type)  # type: ignore
    # Case of a union of types (including optional).
    if origin is Union:
        for cls in typing.get_args(field.type):
            try:
                return _instantiate(cls)
            except TypeError:
                pass
        raise TypeError(
            f"Failed to instantiate {field.name} using type constructors "
            f"{typing.get_args(field.type)}."
        )
    raise TypeError(
        f"Unsupported field type: {field.type} with origin {origin}."
    )


@dataclasses.dataclass
class TomlConfig:
    """Base class to define TOML-parsable configuration containers.

    This class aims at wrapping multiple, possibly optional, sets of
    hyper-parameters, each of which is specified through a dedicated
    class, dataclass, or base python type. The use of some type-hint
    annotations is supported: List, Tuple, Optional and Union may be
    used as long as they are annotated with concrete types.

    It also enables parsing these configuration from a TOML file, as
    well as instantiating from Python objects, possibly using field-
    wise parsers to convert inputs into the desired type.

    Instantiation classmethods
    --------------------------
    from_toml:
        Instantiate by parsing a TOML configuration file.
    from_params:
        Instantiate by parsing inputs dicts (or objects).
    """

    autofill_fields: ClassVar[Set[str]] = set()
    """Class attribute listing names of auto-fill fields.

    The listed fields do not have a formal default value, but one is
    dynamically created upon parsing other fields. As a consequence,
    they may safely been ignored in TOML files or input dict params.
    """

    @classmethod
    def from_params(
        cls,
        **kwargs: Any,
    ) -> Self:
        """Instantiate a structured configuration from input keyword arguments.

        The input keyword arguments should match this class's fields' names.
        For each and every dataclass field of this class:

        - If unprovided, set the argument to None.
        - If a `parse_{field.name}` method exists, use that method.
        - Else, use the `default_parser` method.

        Notes
        -----
        - If a field supports being None, not passing a kwarg for it will
          by default result in setting it to None.
        - If a field has a default value and does not support being None,
          not passing a kwarg for it will by default result in using its
          default value.
        - If a field both has a default value and supports being None, it
          may be preferrable to pass an empty dict kwarg so as to use the
          field's default value rather than a non-default None value.
        - The former remarks hold up when the field does not benefit from
          a specific parser (that may implement arbitrary processing).

        Raises
        ------
        RuntimeError
            In case a field failed to be instantiated using the input key-
            word argument (or None value resulting from the lack thereof).

        Warns
        -----
        UserWarning
            In case some keyword arguments are unused due to the lack of a
            corresponding dataclass field.
        """
        fields: Dict[str, Any] = {}
        # Look up expected kwargs and parse them.
        for field in dataclasses.fields(cls):
            parser = getattr(cls, f"parse_{field.name}", cls.default_parser)
            inputs = kwargs.pop(field.name, None)
            try:
                fields[field.name] = parser(field, inputs)
            except Exception as exc:  # pylint: disable=broad-except
                raise RuntimeError(
                    f"Failed to parse '{field.name}' field: {exc}"
                ) from exc
        # Warn about unused keyword arguments.
        for key in kwargs:
            warnings.warn(
                f"Unsupported keyword argument in {cls.__name__}.from_params: "
                f"'{key}'. This argument was ignored.",
                category=RuntimeWarning,
                stacklevel=2,
            )
        return cls(**fields)

    @classmethod
    def default_parser(
        cls,
        field: dataclasses.Field[T],
        inputs: Union[str, Dict[str, Any], T, None],
    ) -> Any:
        """Default method to instantiate a field from python inputs.

        Parameters
        ----------
        field: dataclasses.Field[<T>]
            Field that is being instantiated.
        inputs: str or dict or <T> or None
            Provided inputs to instantiate the field.
            - If valid as per `field.type`, return inputs.
            - If None (and None is invalid), return the field's default value.
            - If dict, treat them as kwargs to the field's type constructor.
            - If str, treat as the path to a TOML file specifying the object.

        Notes
        -----
        If `inputs` is str and treated as the path to a TOML file,
        it will be parsed in one of the following ways:

        - Call `field.type.from_toml` if `field.type` is a TomlConfig.
        - Use the file's `field.name` section as kwargs, if it exists.
        - Use the entire file's contents as kwargs otherwise.

        Raises
        ------
        TypeError
            If instantiation failed, for any reason.

        Returns
        -------
        object: <T>
            Instantiated object that matches the field's specifications.
        """
        # Case of valid inputs: return them as-is (including valid None).
        if _isinstance_generic(
            inputs,
            field.type,  # type: ignore
        ):  # see function's notes
            return inputs
        # Case of None inputs: return default value if any, else raise.
        if inputs is None:
            if field.default is not dataclasses.MISSING:
                return field.default
            if field.default_factory is not dataclasses.MISSING:
                return field.default_factory()
            raise TypeError(
                f"Field '{field.name}' does not provide a default value."
            )
        # Case of str inputs poiting to a file: try parsing it.
        if isinstance(inputs, str) and os.path.isfile(inputs):
            try:
                with open(inputs, "rb") as file:
                    config = tomllib.load(file, parse_float=_parse_float)
            except tomllib.TOMLDecodeError as exc:
                raise TypeError(
                    f"Field {field.name}: could not parse secondary TOML file."
                ) from exc
            # Look for a subsection, otherwise keep the entire file.
            section = config.get(field.name, config)
            # Recursively call this parser.
            return cls.default_parser(field, section)
        # Case of dict inputs: try instantiating the target type.
        if isinstance(inputs, dict):
            return _instantiate_field(field, **inputs)
        # Otherwise, raise a TypeError.
        raise TypeError(f"Failed to parse inputs for field {field.name}.")

    @classmethod
    def from_toml(
        cls,
        path: str,
        warn_user: bool = True,
        use_section: Optional[str] = None,
        section_fail_ok: bool = False,
    ) -> Self:
        """Parse a structured configuration from a TOML file.

        The parsed TOML configuration file should be organized into sections
        that are named after this class's fields, and provide parameters to
        be parsed by the field's associated dataclass.

        Notes
        -----
        - Sections for fields that have default values may be missing.
        - Parameters with default values may also be missing.
        - All None values should be written as nan ones, as TOML does
          not have a null data type.

        Parameters
        ----------
        path: str
            Path to a TOML configuration file, that provides with the
            hyper-parameters making up for the FL "run" configuration.
        warn_user: bool, default=True
            Boolean indicating whether to raise a warning when some
            fields are unused. Useful for cases where unused fields
            are expected, e.g. in declearn-quickrun mode.
        use_section: optional(str), default=None
            If not None, points to a specific section of the TOML that
            should be used, rather than the whole file. Useful to parse
            orchestrating TOML files, e.g. in declearn-quickrun mode.
        section_fail_ok: bool, default=False
            If True, allow the section specified in use_section to be
            missing from the TOML file without raising an Error.

        Raises
        ------
        RuntimeError
            If parsing fails, whether due to misformatting of the TOML
            file, presence of undue parameters, or absence of required
            ones.

        Warns
        -----
        UserWarning
            In case some sections of the TOML file are unused due to the
            lack of a corresponding dataclass field.
        """
        # Parse the TOML configuration file.
        try:
            with open(path, "rb") as file:
                config = tomllib.load(file, parse_float=_parse_float)
        except tomllib.TOMLDecodeError as exc:
            raise RuntimeError(
                "Failed to parse the TOML configuration file."
            ) from exc
        # Look for expected config sections in the parsed TOML file.
        if isinstance(use_section, str):
            try:
                config = config[use_section]
            except KeyError as exc:
                if not section_fail_ok:
                    raise KeyError("Specified section not found") from exc
        params: Dict[str, Any] = {}
        for field in dataclasses.fields(cls):
            # Case when the section is provided: set it up for parsing.
            if field.name in config:
                params[field.name] = config.pop(field.name)
            # Case when the section is missing: raise if it is required.
            elif (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
                and field.name not in cls.autofill_fields
            ):
                raise RuntimeError(
                    "Missing section in the TOML configuration file: "
                    f"'{field.name}'.",
                )
        # Warn about remaining (unused) config sections.
        if warn_user:
            for name in config:
                warnings.warn(
                    f"Unsupported section encountered in {path} TOML file: "
                    f"'{name}'. This section will be ignored.",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
        # Finally, instantiate the FLConfig container.
        return cls.from_params(**params)
