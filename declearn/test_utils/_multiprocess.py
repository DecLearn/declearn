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

"""Utils to run concurrent routines parallelly using multiprocessing."""

import multiprocessing as mp
from typing import Any, Callable, List, Optional, Tuple


__all__ = [
    "run_as_processes",
]


def run_as_processes(
    *routines: Tuple[Callable[..., Any], Tuple[Any, ...]]
) -> List[Optional[int]]:
    """Run coroutines concurrently within individual processes.

    Parameters
    ----------
    *routines: tuple(function, tuple(any, ...))
        Sequence of routines that need running concurrently,
        each formatted as a 2-elements tuple containing the
        function to run, and a tuple storing its arguments.

    Returns
    -------
    exitcodes: list[int]
        List of exitcodes of the processes wrapping the routines.
        If all codes are zero, then all functions ran properly.
    """
    # Wrap routines as individual processes and run them concurrently.
    processes = [mp.Process(target=func, args=args) for func, args in routines]
    try:
        # Start all processes.
        for process in processes:
            process.start()
        # Regularly check for any failed process and exit if so.
        while any(process.is_alive() for process in processes):
            if any(process.exitcode for process in processes):
                break
            processes[0].join(timeout=1)
    # Ensure not to leave processes running in the background.
    finally:
        for process in processes:
            process.terminate()
    # Return processes' exitcodes.
    return [process.exitcode for process in processes]
