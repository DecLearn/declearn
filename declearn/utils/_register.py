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

"""Generic types-registration system backing some (de)serialization utils."""

import functools
from typing import Dict, Optional, Tuple, Type


__all__ = [
    "access_registered",
    "access_registration_info",
    "access_types_mapping",
    "create_types_registry",
    "register_type",
]


REGISTRIES = {}  # type: Dict[str, TypesRegistry]


class TypesRegistry:
    """Class wrapping a dict registering type classes under str names."""

    def __init__(self, name: str, base: Type) -> None:
        """Instantiate the TypesRegistry.

        Parameters
        ----------
        name: str
            Name of the registry (used to document raised exceptions).
        base: type
            Base class that registered entries should inherit from.
        """
        self.name = name
        self.base = base
        self._reg = {}  # type: Dict[str, Type]

    def get_mapping(self) -> Dict[str, Type]:
        """Return a copy of the mapping managed by this TypesRegistry.

        Returns
        -------
        mapping: Dict[str, type]
            `{name: type}` dict mapping of registered types.
        """
        return self._reg.copy()

    def register(
        self,
        cls: Type,
        name: Optional[str] = None,
        repl: bool = False,
    ) -> None:
        """Add a (name, cls) entry to the registry.

        Parameters
        ----------
        cls: type
            Class that is to be registered.
            Must inherit `self.base`. If not, raise a TypeError.
        name: str or None, default=None
            Name under which the type should be registered, and
            hence retrievable from. If None, use `cls.__name__`.
        repl: bool, default=False
            Whether to overwrite the entry if it already exists.
            If False, raise a `KeyError` if the entry exists.
        """
        if name is None:
            name = cls.__name__
        # Check entry validity.
        if (name in self._reg) and (not repl):
            raise KeyError(f"Name '{name}' has already been registered.")
        if not issubclass(cls, self.base):
            raise TypeError(
                f"'{cls.__name__}' is not a '{self.base.__name__}' subclass."
            )
        # Register entry.
        self._reg[name] = cls

    def access(
        self,
        name: str,
    ) -> Type:
        """Access a registered type by its name.

        Parameters
        ----------
        name: str
            Name under which the type is registered.

        Returns
        -------
        cls: type
            Type retrieved from the registry.
        """
        if name not in self._reg:
            raise KeyError(f"No '{name}' entry under '{self.name}' registry.")
        return self._reg[name]

    def get_name(
        self,
        cls: Type,
    ) -> str:
        """Return the name under which a type has been registered.

        Parameters
        ----------
        cls: type
            Registered type, the storage name of which to retrive.

        Returns
        -------
        name: str
            Name under which the type is registered.
        """
        for name, rtype in self._reg.items():
            if rtype is cls:
                return name
        raise KeyError(
            f"Type '{cls.__name__}' not found in '{self.name}' registry."
        )


def create_types_registry(
    base: Optional[Type] = None,
    name: Optional[str] = None,
) -> Type:
    """Create a TypesRegistry backing generic (de)serialization utils.

    Note: this function may either be used to create a registy with
          an existing type as base through functional syntax, or be
          placed as a decorator for class-defining code

    Parameters
    ----------
    base: type or None, default=None
        Base class that registered entries should inherit from.
        If None, return a class decorator.
    name: str or None, default=None
        Name of the registry, used to register or access classes
        using the generic `register_type` and `access_registered`
        utility functions.
        If None, use `base.__name__`

    Returns
    -------
    base: type
        The input `base`; hence this function may be used as
        a decorator to register classes in the source code.
    """
    # Case when the function is being used as a class decorator.
    if base is None:
        decorator = functools.partial(create_types_registry, name=name)
        return decorator  # type: ignore
    # Create the types registry, after checking it does not already exist.
    name = base.__name__ if name is None else name
    if name in REGISTRIES:
        raise KeyError(f"TypesRegistry '{name}' already exists.")
    REGISTRIES[name] = TypesRegistry(name, base)
    return base


def register_type(
    cls: Optional[Type] = None,
    name: Optional[str] = None,
    group: Optional[str] = None,
) -> Type:
    """Register a class in a registry, to ease its (de)serialization.

    Note: this function may either be used to register an existing
          type through functional syntax, or placed as a decorator
          for class-defining code

    Parameters
    ----------
    cls: type or None, default=None
        Class that is to be registered.
        If None, return a class decorator.
    name: str or None, default=None
        Name under which the type should be registered, and
        hence retrievable from. If None, use `cls.__name__`.
    group: str or None, default=None
        Name of the TypesRegistry to which the class should
        be added (created using `create_types_registry`).
        If None, use the first-found existing registry that
        has a compatible base type, or raise a TypeError.

    Returns
    -------
    cls: type
        The input `cls`; hence this function may be used as
        a decorator to register classes in the source code.
    """
    # Case when the function is being used as a class decorator.
    if cls is None:
        decorator = functools.partial(register_type, name=name, group=group)
        return decorator  # type: ignore
    # Optionnally infer the registry to use. Otherwise, check existence.
    if group is None:
        for key, reg in REGISTRIES.items():
            if issubclass(cls, reg.base):
                group = key
                break
        else:
            raise TypeError("Could not infer registration group.")
    elif group not in REGISTRIES:
        raise KeyError(f"Type registry '{group}' does not exist.")
    # Register the type in the target registry.
    REGISTRIES[group].register(cls, name)
    # Return the input type (enabling use as a decorator).
    return cls


def access_registered(
    name: str,
    group: Optional[str] = None,
) -> Type:
    """Access a registered type by its name.

    Parameters
    ----------
    name: str
        Name under which the type is registered.
    group: str or None, default=None
        Name of the TypesRegistry under which the type is stored.
        If None, look for the name in each and every registry and
        return the first-found match or raise a KeyError.

    Returns
    -------
    cls: type
        Type retrieved from a types registry.

    Raises
    ------
    KeyError:
        If no registered type matching the input parameters is found.
    """
    # If group is unspecified, look the name up in each and every registry.
    if group is None:
        for reg in REGISTRIES.values():
            try:
                return reg.access(name)
            except KeyError:
                continue
        raise KeyError(f"No '{name}' entry under any types registry.")
    # Otherwise, look up the registry, then access the target type from it.
    if group not in REGISTRIES:
        raise KeyError(f"Type registry '{group}' does not exist.")
    return REGISTRIES[group].access(name)


def access_registration_info(
    cls: Type,
    group: Optional[str] = None,
) -> Tuple[str, str]:
    """Access a registered type's storage name and belonging group.

    Parameters
    ----------
    cls: str
        Registered type, the storage name of which to retrive.
    group: str or None, default=None
        Name of the TypesRegistry under which the type is stored.
        If None, look for the type in each and every registry and
        return the first-found match or raise a KeyError.

    Returns
    -------
    name: str
        Name under which the type is registered.
    group: str
        Name of the TypesRegistry in which the type is registered.

    Raises
    ------
    KeyError:
        If the provided information does not match a registered type.
    """
    # If group is unspecified, look the type up in each and every registry.
    if group is None:
        for grp, reg in REGISTRIES.items():
            try:
                return reg.get_name(cls), grp
            except KeyError:
                continue
        raise KeyError(
            f"Type '{cls.__name__}' not found under any types registry."
        )
    # Otherwise, look up the registry, then get the target name from it.
    if group not in REGISTRIES:
        raise KeyError(f"Type registry '{group}' does not exist.")
    return REGISTRIES[group].get_name(cls), group


def access_types_mapping(
    group: str,
) -> Dict[str, Type]:
    """Return a copy of the `{name: type}` mapping of a given group.

    Parameters
    ----------
    group: str
        Name of a TypesRegistry, the mapping from which to return.

    Returns
    -------
    mapping: Dict[str, type]
        `{name: type}` dict mapping of registered types.
        Note that this is a copy of the actual registry.

    Raises
    ------
    KeyError:
        If the `group` types registry does not exist.
    """
    if group not in REGISTRIES:
        raise KeyError(f"Type registry '{group}' does not exist.")
    return REGISTRIES[group].get_mapping()
