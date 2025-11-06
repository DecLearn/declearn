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

"""Utils to define a computation device policy.

This private submodule defines:

- A dataclass defining a standard to hold a device-selection policy.
- A private global variable holding the current package-wise device policy.
- A public pair of functions acting as a getter and a setter for that variable.
"""

import dataclasses
from typing import Optional

__all__ = [
    "DevicePolicy",
    "get_device_policy",
    "set_device_policy",
]


@dataclasses.dataclass
class DevicePolicy:
    """Dataclass to store parameters defining a device-selection policy.

    This class merely defines a shared language of keyword-arguments to
    define whether to back computations on CPU or on a GPU device.

    It is meant to be instantiated as a global variable that holds that
    information, and can be accessed by framework-specific backend code
    so as to take the required steps towards implementing that policy.

    To access or update the current global DevicePolicy, please use the
    getter and setter functions: `declearn.utils.get_device_policy` and
    `declearn.utils.set_device_policy`.

    Attributes
    ----------
    gpu: bool
        Whether to use a GPU device rather than the CPU one to back data
        and computations. If no GPU is available, use CPU with a warning.
    idx: int or None
        Optional index of the GPU device to use.
        If None, select one arbitrarily.
        If this index exceeds the number of available GPUs, select one
        arbitrarily, with a warning.
    """

    gpu: bool
    idx: Optional[int]

    def __post_init__(self) -> None:
        if not isinstance(self.gpu, bool):
            raise TypeError(
                f"DevicePolicy 'gpu' should be a bool, not '{type(self.gpu)}'."
            )
        if not (self.idx is None or isinstance(self.idx, int)):
            raise TypeError(
                "DevicePolicy 'idx' should be None or an int, not "
                f"'{type(self.idx)}'."
            )


DEVICE_POLICY = DevicePolicy(  # pylint: disable=[invalid-name]
    gpu=True, idx=None
)


def get_device_policy() -> DevicePolicy:
    """Return a copy of the current global device policy.

    This method is meant to be used:

    - By end-users that wish to check the current device policy.
    - By the backend code of framework-specific objects so as to
      take the required steps towards implementing that policy.

    To update the current policy, use `declearn.utils.set_device_policy`.

    Returns
    -------
    policy: DevicePolicy
        DevicePolicy dataclass instance, wrapping parameters that specify
        the device policy to be enforced by Model and Vector to properly
        place data and computations.
    """
    return DevicePolicy(**dataclasses.asdict(DEVICE_POLICY))


def set_device_policy(
    gpu: bool,
    idx: Optional[int] = None,
) -> None:
    """Update the current global device policy.

    To access the current policy, use `declearn.utils.get_device_policy`.

    Parameters
    ----------
    gpu: bool
        Whether to use a GPU device rather than the CPU one to back data
        and computations. If no GPU is available, use CPU with a warning.
    idx: int or None, default=None
        Optional index of the GPU device to use.
        If this index exceeds the number of available GPUs, select one
        arbitrarily, with a warning.
    """
    # Using a global statement to have a proper setter to a private variable.
    global DEVICE_POLICY  # pylint: disable=global-statement
    DEVICE_POLICY = DevicePolicy(gpu, idx)
