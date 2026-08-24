# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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

"""Common format-agnostic tools for lazy serialization support of
Declearn objects.
"""

import logging
import warnings
from importlib import import_module
from typing import Dict

_LAZY_SERIAL_REGISTRY: Dict[str, str] = {}
"""Registry mapping a DecLearn type name to its module (the import of which
will trigger the actual registration for serialization).
"""

logger = logging.getLogger(__name__)


def add_lazy_serial_support(type_name: str, module: str) -> None:
    """Add lazy serialization support for the type `type_name`.

    It consists in storing in a registry the `type_name` mapped to its
    `module`, in order to know which module we need to manually import to
    trigger the actual serialization support.

    Parameters
    ----------
    type_name : str
        Name of the type to be lazy-registered.
    module : str
        Name of the module containing the type `type_name`,
        e.g. `declearn.model.torch`.
    """
    if type_name in _LAZY_SERIAL_REGISTRY:
        warnings.warn(
            f"Type '{type_name}' has already been lazy-registered",
            stacklevel=2,
        )
    _LAZY_SERIAL_REGISTRY[type_name] = module


def import_module_of(type_name: str) -> None:
    """Import the module containing `type_name`, to trigger its
    serialization support.

    A call to this function will manually import the module containing
    `type_name` (they are stored together in the _LAZY_SERIAL_REGISTRY),
    consequently triggering the serialization support for this type.

    Warning
    -------
    This function assumes that importing the associated module is sufficient
    to trigger a call to `add_serialization_support` for the concerned type,
    which is normally the case for serializable types in optional modules.
    """
    module = _LAZY_SERIAL_REGISTRY[type_name]
    logger.info(
        f"Manually importing '{module}' to trigger serialization "
        f"support for '{type_name}'."
    )
    import_module(module)
