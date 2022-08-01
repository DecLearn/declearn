# coding: utf-8

"""Generic types-registration system backing some (de)serialization utils."""

import functools
from typing import Dict, Type, Optional


__all__ = [
    'access_registered',
    'create_types_registry',
    'register_type',
]


REGISTRIES = {}  # type: Dict[str, TypesRegistry]


class TypesRegistry:
    """Class wrapping a dict registering type classes under str names."""

    def __init__(
            self,
            name: str,
            base: Type
        ) -> None:
        """Instantiate the TypesRegistry.

        Arguments:
        ---------
        name: str
            Name of the registry (used to document raised exceptions).
        base: type
            Base class that registered entries should inherit from.
        """
        self.name = name
        self.base = base
        self._reg = {}  # type: Dict[str, Type]

    def register(
            self,
            cls: Type,
            name: Optional[str] = None,
            repl: bool = False
        ) -> None:
        """Add a (name, cls) entry to the registry.

        Arguments:
        ---------
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
            name: str
        ) -> Type:
        """Access a registered type by its name.

        Arguments:
        ---------
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


def create_types_registry(
        name: str,
        base: Type
    ) -> None:
    """Create a TypesRegistry backing generic (de)serialization utils.

    Arguments:
    ---------
    name: str
        Name of the registry, used to register or access classes
        using the generic `register_type` and `access_registered`
        utility functions.
    base: type
        Base class that registered entries should inherit from.
    """
    if name in REGISTRIES:
        raise KeyError(f"TypesRegistry '{name}' already exists.")
    REGISTRIES[name] = TypesRegistry(name, base)


def register_type(
        cls: Optional[Type] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
    ) -> Type:
    """Register a class in a registry, to ease its (de)serialization.

    Note: this function may either be used to register an existing
          type through functional syntax, or placed as a decorator
          for class-defining code

    Arguments:
    ---------
    cls: type of None, default=None
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
    #
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
        group: Optional[str] = None
    ) -> Type:
    """Access a registered type by its name.

    Arguments:
    ---------
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
