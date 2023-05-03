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

import functools
import multiprocessing as mp
import sys
import traceback
from queue import Queue
from typing import Any, Callable, Dict, List, Tuple, Union

__all__ = [
    "run_as_processes",
]


def run_as_processes(
    *routines: Tuple[Callable[..., Any], Tuple[Any, ...]],
    auto_stop: bool = True,
) -> Tuple[bool, List[Union[Any, RuntimeError]]]:
    """Run coroutines concurrently within individual processes.

    Parameters
    ----------
    *routines: tuple(function, tuple(any, ...))
        Sequence of routines that need running concurrently,
        each formatted as a 2-elements tuple containing the
        function to run, and a tuple storing its arguments.
    auto_stop: bool, default=True
        Whether to automatically interrupt all running routines
        as soon as one failed and raised an exception. This can
        avoid infinite runtime (e.g. if one awaits for a failed
        routine to send information), but may also prevent some
        exceptions from being caught due to the early stopping
        of routines that would have failed later. Hence it may
        be disabled in contexts where it is interesting to wait
        for all routines to fail rather than assume that they
        are co-dependent.

    Returns
    -------
    success: bool
        Whether all routines were run without raising an exception.
    outputs: list[RuntimeError or Any]
        List of routine-wise output value or RuntimeError exception
        that either wraps an actual exception and its traceback, or
        indicates that the process was interrupted while running.
    """
    # Wrap routines into named processes and set up exceptions catching.
    queue = (
        mp.Manager().Queue()
    )  # type: Queue[Tuple[str, Union[Any, RuntimeError]]]
    names = []  # type: List[str]
    count = {}  # type: Dict[str, int]
    processes = []  # type: List[mp.Process]
    for func, args in routines:
        name = func.__name__
        nidx = count[name] = count.get(name, 0) + 1
        name = f"{name}-{nidx}"
        func = add_exception_catching(func, queue, name)
        names.append(name)
        processes.append(mp.Process(target=func, args=args, name=name))
    # Run the processes concurrently.
    try:
        # Start all processes.
        for process in processes:
            process.start()
        # Regularly check for any failed process and exit if so.
        while any(process.is_alive() for process in processes):
            if auto_stop and any(process.exitcode for process in processes):
                break
            # Wait for at most 1 second on the first alive process.
            for process in processes:
                if process.is_alive():
                    process.join(timeout=1)
                    break
    # Ensure not to leave processes running in the background.
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
    # Return success flag and re-ordered outputs and exceptions.
    success = all(process.exitcode == 0 for process in processes)
    dequeue = dict([queue.get_nowait() for _ in range(queue.qsize())])
    int_err = RuntimeError("Process was interrupted while running.")
    outputs = [dequeue.get(name, int_err) for name in names]
    return success, outputs


def add_exception_catching(
    func: Callable[..., Any],
    queue: Queue,
    name: str,
) -> Callable[..., Any]:
    """Wrap a function to catch exceptions and put them in a Queue."""
    return functools.partial(wrapped, func=func, queue=queue, name=name)


def wrapped(
    *args: Any,
    func: Callable[..., Any],
    queue: Queue,
    name: str,
) -> Any:
    """Call the wrapped function and catch exceptions or results."""
    try:
        result = func(*args)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        err = RuntimeError(
            f"Exception of type {type(exc)} occurred:\n"
            "".join(traceback.format_exception(type(exc), exc, tb=None))
        )  # future: `traceback.format_exception(exc)` (py >=3.10)
        queue.put((name, err))
        sys.exit(1)
    else:
        queue.put((name, result))
        sys.exit(0)
