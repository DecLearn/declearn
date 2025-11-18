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

"""Utils to run concurrent routines parallelly using multiprocessing."""

import functools
import multiprocessing as mp
import sys
import traceback
from queue import Queue
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

__all__ = [
    "run_as_processes",
]


def run_as_processes(
    *routines: Union[
        Tuple[Callable[..., Any], Tuple[Any, ...]],
        Tuple[Callable[..., Any], Dict[str, Any]],
        Tuple[Callable[..., Any], Tuple[Any, ...], Dict[str, Any]],
    ],
    auto_stop: bool = True,
) -> Tuple[bool, List[Union[Any, RuntimeError]]]:
    """Run coroutines concurrently within individual processes.

    Parameters
    ----------
    *routines: tuple(function, tuple(any, ...))
        Sequence of routines that need running concurrently, each
        formatted as either:
        - a 3-elements tuple containing the function to run,
            a tuple of positional args and a dict of kwargs.
        - a 2-elements tuple containing the function to run,
            and a tuple storing its (positional) arguments.
        - a 2-elements tuple containing the function to run,
            and a dict storing its keyword arguments.
    auto_stop: bool, default=True
        Whether to automatically interrupt all running routines as
        soon as one failed and raised an exception. This can avoid
        infinite runtime (e.g. if one awaits for a failed routine
        to send information), but may also prevent some exceptions
        from being caught due to the early stopping of routines that
        would have failed later. Hence it may be disabled in contexts
        where it is interesting to wait for all routines to fail rather
        than assume that they are co-dependent.

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
    queue: Queue[Tuple[str, Union[Any, RuntimeError]]] = mp.Manager().Queue()
    processes, names = prepare_routine_processes(routines, queue)
    # Run the processes concurrently.
    run_processes(processes, auto_stop)
    # Return success flag and re-ordered outputs and exceptions.
    success = all(process.exitcode == 0 for process in processes)
    dequeue = dict([queue.get_nowait() for _ in range(queue.qsize())])
    int_err = RuntimeError("Process was interrupted while running.")
    outputs = [dequeue.get(name, int_err) for name in names]
    return success, outputs


def prepare_routine_processes(
    routines: Iterable[
        Union[
            Tuple[Callable[..., Any], Tuple[Any, ...]],
            Tuple[Callable[..., Any], Dict[str, Any]],
            Tuple[Callable[..., Any], Tuple[Any, ...], Dict[str, Any]],
        ]
    ],
    queue: Queue[Tuple[str, Union[Any, RuntimeError]]],
) -> Tuple[List[mp.Process], List[str]]:
    """Wrap up routines into named unstarted processes.

    Parameters
    ----------
    routines:
        Iterators of (function, args) tuples to wrap as processes.
    queue:
        Queue where to put the routines' return value or raised exception
        (always wrapped into a RuntimeError), together with their name.

    Raises
    ------
    TypeError
        If the inputs do not match the expected type specifications.

    Returns
    -------
    processes:
        List of `multiprocessing.Process` instances wrapping `routines`.
    names:
        List of names identifying the processes (used for results collection).
    """
    names: List[str] = []
    count: Dict[str, int] = {}
    processes: List[mp.Process] = []
    for routine in routines:
        func, args, kwargs = parse_routine_specification(routine)
        name = func.__name__
        nidx = count[name] = count.get(name, 0) + 1
        name = f"{name}-{nidx}"
        func = add_exception_catching(func, queue, name)
        names.append(name)
        processes.append(
            mp.Process(target=func, args=args, kwargs=kwargs, name=name)
        )
    return processes, names


def parse_routine_specification(
    routine: Union[
        Tuple[Callable[..., Any], Tuple[Any, ...]],
        Tuple[Callable[..., Any], Dict[str, Any]],
        Tuple[Callable[..., Any], Tuple[Any, ...], Dict[str, Any]],
    ],
) -> Tuple[Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]:
    """Type-check and unpack a given routine specification.

    Raises
    ------
    TypeError
        If the inputs do not match the expected type specifications.

    Returns
    -------
    func:
        Callable to wrap as a process.
    args:
        Tuple of positional arguments to `func`. May be empty.
    kwargs:
        Dict of keyword arguments to `func`. May be empty.
    """
    # Type-check the overall input.
    if not (isinstance(routine, (tuple, list)) and (len(routine) in (2, 3))):
        raise TypeError(
            "Received an unproper routine specification: should be a 2- "
            "or 3-element tuple."
        )
    # Check that the first argument is callable.
    func = routine[0]
    if not callable(func):
        raise TypeError(
            "The first argument of a routine specification should be callable."
        )
    # Case of a 2-elements tuple: may be (func, args) or (func, kwargs).
    if len(routine) == 2:
        if isinstance(routine[1], tuple):
            args = routine[1]
            kwargs = {}
        elif isinstance(routine[1], dict):
            args = tuple()
            kwargs = routine[1]
        else:
            raise TypeError(
                "Received an unproper routine specification: 2nd element "
                f"should be a tuple or dict, not '{type(routine[1])}'."
            )
    # Case of a 3-elements tuple: should be (func, args, kwargs).
    else:
        args = routine[1]  # type: ignore  # verified below
        kwargs = routine[2]  # type: ignore  # verified below
        if not (isinstance(args, tuple) and isinstance(kwargs, dict)):
            raise TypeError(
                "Received an unproper routine specification: 2nd and 3rd "
                f"elements should be a tuple and a dict, not '{type(args)}'"
                f" and '{type(kwargs)}'."
            )
    return func, args, kwargs


def add_exception_catching(
    func: Callable[..., Any],
    queue: Queue[Tuple[str, Union[Any, RuntimeError]]],
    name: str,
) -> Callable[..., Any]:
    """Wrap a function to catch exceptions and put them in a Queue."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        """Call the wrapped function and queue exceptions or results."""
        nonlocal name, queue
        try:
            result = func(*args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            err = RuntimeError(
                f"Exception of type {type(exc)} occurred:\n"
                + "".join(traceback.format_exception(exc))
            )
            queue.put((name, err))
            sys.exit(1)
        else:
            queue.put((name, result))
        sys.exit(0)

    return wrapped


def run_processes(
    processes: List[mp.Process],
    auto_stop: bool,
) -> None:
    """Run parallel processes, optionally interrupting all if any fails."""
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
