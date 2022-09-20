# coding: utf-8

"""Generic tools to (de-)serialize custom declearn objects to and from JSON."""

import dataclasses
import json
from typing import Any, Dict, Optional, Type, Union

from typing_extensions import (
    Self,       # future: import from typing (Py>=3.11)
    TypedDict,  # future: import from typing (Py>=3.8)
)

from declearn.utils._register import (
    access_registered, access_registration_info
)
from declearn.utils._json import json_pack, json_unpack
from declearn.typing import SupportsConfig


__all__ = [
    'ObjectConfig',
    'deserialize_object',
    'serialize_object',
]


ObjectConfigDict = TypedDict(
    "ObjectConfigDict",
    {"name": str, "group": Optional[str], "config": Dict[str, Any]}
)


@dataclasses.dataclass
class ObjectConfig:
    """Dataclass to wrap a JSON-serializable object configuration.

    Attributes
    ----------
    name: str
        Key to retrieve the object's type constructor, typically from
        registered types (see `declearn.utils.access_registered`).
    group: str or None
        Optional name of the group under which the object's type is
        registered (see `declearn.utils.access_registered`).
    config: dict[str, any]
        JSON-serializable dict containing the object's config, that
        enables recreating it using `type(obj).from_config(config)`.
    """

    name: str
    group: Optional[str]
    config: Dict[str, Any]

    def to_dict(
            self
        ) -> ObjectConfigDict:
        """Return a dict representation of this ObjectConfig."""
        return dataclasses.asdict(self)  # type: ignore

    def to_json(
            self,
            path: str
        ) -> None:
        """Save this ObjectConfig to a JSON file."""
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2, default=json_pack)

    @classmethod
    def from_json(
            cls,
            path: str
        ) -> Self:  # type: ignore
        """Restore an ObjectConfig from a JSON file."""
        with open(path, "r", encoding="utf-8") as file:
            config = json.load(file, object_hook=json_unpack)
        return cls(**config)


def serialize_object(
        obj: SupportsConfig,
        group: Optional[str] = None,
        allow_unregistered: bool = False,
    ) -> ObjectConfig:
    """Return a ObjectConfig serialization of an object.

    This function is the counterpart to `declearn.utils.deserialize_object`.

    Parameters
    ----------
    obj: object
        Object that needs serialization. To be valid, the object must:
        * implement the `get_config` and `from_config` (class)methods
        * belong to a registered type, unless `allow_unregistered=True`
          (i.e. a type that has been passed to or decorated with the
          `declearn.utils.register_type` function)
    group: str or None, default=None
        Optional name of the group under which the object's type was
        registered. If None, the type will be looked for in each and
        every exiting group, and the first match will be used.
    allow_unregistered: bool, default=False
        Whether to allow serializing objects that do not belong to a
        registered type. If true, a mapping between the type's name
        and its class will need to be passed as part of `custom` to
        the deserialization function (see `deserialize_object`).

    Returns
    -------
    config: ObjectConfig
        A JSON-serializable dataclass wrapper containing all required
        information to recreate the object, e.g. from a config file.

    Raises
    ------
    KeyError
        If `obj`'s type is not registered and `allow_unregistered=False`.
    """
    try:
        name, group = access_registration_info(type(obj), group)
    except KeyError as exception:
        if allow_unregistered:
            name = type(obj).__name__
            group = None
        else:
            raise exception
    config = obj.get_config()
    return ObjectConfig(name, group, config)


def deserialize_object(
        config: Union[str, ObjectConfig, ObjectConfigDict],
        custom: Optional[Dict[str, Type[SupportsConfig]]] = None,
    ) -> SupportsConfig:
    """Return an object from a ObjectConfig serialization or JSON file.

    This function is the counterpart to `declearn.utils.serialize_object`.

    Parameters
    ----------
    config: ObjectConfig or str
        Either an ObjectConfig object, the dict representation of one,
        or the path to a JSON file that stores an ObjectConfig dump.
    custom: dict[str, object] or None, default=None
        Optional dict providing with a {name: type} mapping to enable
        deserializing user-defined types that have not been registered
        using `declearn.utils.register_type`.

    Returns
    -------
    obj: object
        An object instantiated from the provided configuration.

    Raises
    ------
    TypeError
        If `config` is not or cannot be transformed into an ObjectConfig.
    KeyError
        If `config` specifies an object type that cannot be retrieved.
    """
    if isinstance(config, str):
        config = ObjectConfig.from_json(config)
    elif isinstance(config, dict):
        try:
            config = ObjectConfig(**config)
        except TypeError as exc:
            raise TypeError(
                "deserialize_object received a 'config' dict that does not "
                "conform with the ObjectConfig specification."
            ) from exc
    elif not isinstance(config, ObjectConfig):
        raise TypeError(
            "Unproper type for argument 'config' of deserialize_object: "
            f"'{type(config)}'."
        )
    try:
        cls = access_registered(config.name, config.group)
    except KeyError as exception:
        if custom is None:
            raise exception
        if config.name not in custom:
            msg = f"No custom type provided for name '{config.name}'"
            raise KeyError(msg) from exception
        cls = custom[config.name]
    return cls.from_config(config.config)  # type: ignore
