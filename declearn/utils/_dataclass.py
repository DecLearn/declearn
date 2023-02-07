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

"""Dataclass-generation tools.

These tools are meant to reduce redundant code and ease maintanability
when dataclasses are built to gather / parse / validate parameters so
as to eventually call a given function or build a given object.
"""

# fmt: off
import dataclasses
import inspect
from typing import (
    Any, Callable, Dict, List, Optional, Protocol, Tuple, Type, TypeVar
)
# fmt: on


__all__ = [
    "dataclass_from_func",
    "dataclass_from_init",
]


T_co = TypeVar("T_co", covariant=True)  # typevar to annotate created classes
S = TypeVar("S")  # typevar to annotate input classes in `from_init` function


class DataclassFromFunc(Protocol[T_co]):
    """Protocol for dataclasses generated through `dataclass_from_func`."""

    # protocol; pylint: disable=too-few-public-methods

    def call(self) -> T_co:
        """Call the function using the wrapped parameters."""


class DataclassFromInit(Protocol[T_co]):
    """Protocol for dataclasses generated through `dataclass_from_init`."""

    # protocol; pylint: disable=too-few-public-methods

    def instantiate(self) -> T_co:
        """Instantiate from the wrapped parameters."""


def dataclass_from_func(
    func: Callable[..., S],
    name: Optional[str] = None,
) -> Type[DataclassFromFunc[S]]:
    """Automatically build a dataclass matching a function's signature.

    Parameters
    ----------
    func: callable
        Function, the input signature of which to wrap up as a dataclass.
    name: str or None, default=None
        Name to attach to the returned dataclass.
        If None, use CamelCase-converted `func.__name__` + "Config"
        (e.g. "MyFuncConfig" for a "my_func" input function).

    Returns
    -------
    dataclass: Dataclass-built type
        Dataclass, the fields of which are the input arguments to `func`
        (with *args as a list and **kwargs as a dict), exposing a `call`
        method that triggers calling `func` with the wrapped parameters.
    """
    # Parse the function's signature into dataclass Field instances.
    signature = inspect.signature(func)
    parameters = list(signature.parameters.values())
    fields = _parameters_to_fields(parameters)
    # Make a dataclass out of the former fields.
    if not name:
        name = "".join(w.capitalize() for w in func.__name__.split("_"))
        name += "Config"
    dcls = dataclasses.make_dataclass(name, fields)  # type: Type
    # Bind the dataclass's main and __init__ docstrings.
    docs = f"Dataclass for {func.__name__} instantiation parameters.\n"
    dcls.__doc__ = docs
    dcls.__init__.__doc__ = docs + (func.__doc__ or "").split("\n", 1)[-1]
    # If the signature comprises *args / **kwargs parameters, record it.
    args_field = kwargs_field = None  # type: Optional[str]
    for param in parameters:
        if param.kind is param.VAR_POSITIONAL:
            args_field = param.name
        if param.kind is param.VAR_KEYWORD:
            kwargs_field = param.name
    # Add a method to instantiate from the dataclass.
    r_type = (
        Any
        if signature.return_annotation is signature.empty
        else signature.return_annotation
    )

    def call(self) -> r_type:  # type: ignore
        """Call from the wrapped parameters."""
        nonlocal args_field, kwargs_field
        params = dataclasses.asdict(self)
        args = params.pop(args_field) if args_field else []
        kwargs = params.pop(kwargs_field) if kwargs_field else {}
        return func(*args, **params, **kwargs)

    call.__doc__ = (
        f"Call function {func.__name__} from the wrapped parameters."
    )
    dcls.call = call
    # Return the generated dataclass.
    return dcls


def dataclass_from_init(
    cls: Type[S],
    name: Optional[str] = None,
) -> Type[DataclassFromInit[S]]:
    """Automatically build a dataclass matching a class's init signature.

    Parameters
    ----------
    cls: Type
        Class, the __init__ signature of which to wrap up as a dataclass.
    name: str or None, default=None
        Name to attach to the returned dataclass.
        If None, use `cls.__name__` + "Config"
        (e.g. "MyClassConfig" for a "MyClass" input class).

    Returns
    -------
    dataclass: Dataclass-built type
        Dataclass, the fields of which are the input arguments to the
        `cls.__init__` method (with *args as a list and **kwargs as a
        dict), exposing an `instantiate` method that triggers calling
        `cls(...)` with the wrapped parameters.
    """
    # Parse the class's __init__ signature into dataclass Field instances.
    parameters = list(inspect.signature(cls.__init__).parameters.values())[1:]
    fields = _parameters_to_fields(parameters)
    # Make a dataclass out of the former fields.
    name = name or f"{cls.__name__}Config"
    dcls = dataclasses.make_dataclass(name, fields)  # type: Type
    # Bind the dataclass's main and __init__ docstrings.
    docs = f"Dataclass for {cls.__name__} instantiation parameters.\n"
    dcls.__doc__ = docs
    dcls.__init__.__doc__ = (
        docs + (cls.__init__.__doc__ or "").split("\n", 1)[-1]
    )
    # If the signature comprises *args / **kwargs parameters, record it.
    args_field = kwargs_field = None  # type: Optional[str]
    for param in parameters:
        if param.kind is param.VAR_POSITIONAL:
            args_field = param.name
        if param.kind is param.VAR_KEYWORD:
            kwargs_field = param.name

    # Add a method to instantiate from the dataclass.
    def instantiate(self) -> cls:  # type: ignore
        """Instantiate from the wrapped init parameters."""
        nonlocal args_field, kwargs_field
        params = dataclasses.asdict(self)
        args = params.pop(args_field) if args_field else []
        kwargs = params.pop(kwargs_field) if kwargs_field else {}
        return cls(*args, **params, **kwargs)

    instantiate.__doc__ = (
        f"Instantiate a {cls.__name__} from the wrapped init parameters."
    )
    dcls.instantiate = instantiate
    # Return the generated dataclass.
    return dcls


def _parameters_to_fields(
    params: List[inspect.Parameter],
) -> List[Tuple[str, Type, dataclasses.Field]]:
    """Parse function or method parameters into dataclass fields."""
    fields = []  # type: List[Tuple[str, Type, dataclasses.Field]]
    for param in params:
        # Parse out the parameter's name, annotated type and default value.
        fname = param.name
        ftype = Any if param.annotation is param.empty else param.annotation
        field = dataclasses.field()
        if param.default is not param.empty:
            field.default = param.default
        # Turn *args into a list and **kwargs into a dict.
        if param.kind is param.VAR_POSITIONAL:
            ftype = List[ftype]  # type: ignore
            field.default_factory = list
        elif param.kind is param.VAR_KEYWORD:
            ftype = Dict[str, ftype]  # type: ignore
            field.default_factory = dict
        # Append parsed information to the fields list.
        fields.append((fname, ftype, field))
    return fields
