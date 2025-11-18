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

"""Utils to list available optimizer plug-ins (OptiModule and Regularizer)."""

from typing import Dict, Type

from declearn.optimizer.modules import OptiModule
from declearn.optimizer.regularizers import Regularizer
from declearn.optimizer.schedulers import Scheduler
from declearn.utils import access_types_mapping

__all__ = [
    "list_optim_modules",
    "list_optim_regularizers",
    "list_rate_schedulers",
]


def list_optim_modules() -> Dict[str, Type[OptiModule]]:
    """Return a mapping of registered OptiModule subclasses.

    This function aims at making it easy for end-users to list and access
    all available OptiModule optimizer plug-ins at any given time. The
    returned dict uses unique identifier keys, which may be used to add
    the associated plug-in to a [declearn.optimizer.Optimizer][] without
    going through the fuss of importing and instantiating it manually.

    Note that the mapping will include all declearn-provided plug-ins,
    but also registered plug-ins provided by user or third-party code.

    See also
    --------
    * [declearn.optimizer.modules.OptiModule][]:
        API-defining abstract base class for the OptiModule plug-ins.
    * [declearn.optimizer.list_optim_regularizers][]:
        Counterpart function for Regularizer plug-ins.

    Returns
    -------
    mapping:
        Dictionary mapping unique str identifiers to OptiModule
        class constructors.
    """
    return access_types_mapping("OptiModule")


def list_optim_regularizers() -> Dict[str, Type[Regularizer]]:
    """Return a mapping of registered Regularizer subclasses.

    This function aims at making it easy for end-users to list and access
    all available Regularizer optimizer plug-ins at any given time. The
    returned dict uses unique identifier keys, which may be used to add
    the associated plug-in to a [declearn.optimizer.Optimizer][] without
    going through the fuss of importing and instantiating it manually.

    Note that the mapping will include all declearn-provided plug-ins,
    but also registered plug-ins provided by user or third-party code.

    See also
    --------
    * [declearn.optimizer.regularizers.Regularizer][]:
        API-defining abstract base class for the Regularizer plug-ins.
    * [declearn.optimizer.list_optim_modules][]:
        Counterpart function for OptiModule plug-ins.

    Returns
    -------
    mapping:
        Dictionary mapping unique str identifiers to Regularizer
        class constructors.
    """
    return access_types_mapping("Regularizer")


def list_rate_schedulers() -> Dict[str, Type[Scheduler]]:
    """Return a mapping of registered Scheduler subclasses.

    This function aims at making it easy for end-users to list and access
    all available Scheduler classes at any given time. The returned dict
    uses unique identifier keys, which may be used to use the associated
    scheduler within a [declearn.optimizer.Optimizer][] without going
    through the fuss of importing and instantiating it manually.

    Note that the mapping will include all declearn-provided schedulers,
    but also registered schedulers provided by user or third-party code.

    See also
    --------
    * [declearn.optimizer.schedulers.Scheduler][]:
        API-defining abstract base class for the Scheduler classes.

    Returns
    -------
    mapping:
        Dictionary mapping unique str identifiers to `Scheduler`
        class constructors.
    """
    return access_types_mapping("Scheduler")
