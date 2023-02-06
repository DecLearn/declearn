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

"""Dependency functions for a FL server to process 'data_info'."""

from typing import Any, Dict, Set, NoReturn


from declearn.data_info import aggregate_data_info


__all__ = [
    "AggregationError",
    "aggregate_clients_data_info",
]


class AggregationError(Exception):
    """Custom Exception, used to wrap data-info-aggregation failure info."""

    def __init__(
        self,
        error: str,
        messages: Dict[str, str],
    ) -> None:
        """Instantiate the AggregationError, wrapping error information.

        Parameters
        ----------
        error: str
            General error message reporting on the reasons for which
            client-wise data-info aggregation failed.
        messages: dict[str, str]
            Client-wise error messages reporting on the reason for
            which data-info aggregation failed, formatted as a
            `{client_name: error_msg}` dict.

        Note
        ----
        The `error` and `messages` arguments are available as attributes,
        effectively making this exception a wrapper to pass information.
        """
        super().__init__(error, messages)
        self.error = error
        self.messages = messages

    def __str__(
        self,
    ) -> str:
        return self.error


def aggregate_clients_data_info(
    clients_data_info: Dict[str, Dict[str, Any]],
    required_fields: Set[str],
) -> Dict[str, Any]:
    """Validate and aggregate clients' data-info dictionaries.

    This functions wraps `declearn.data_info.aggregate_data_info`,
    catches KeyError and ValueError and runs further analysis to
    identify their cause and raise a custom AggregationError that
    stores a general error message and client-wise ones in order
    to report on failure causes.

    Parameters
    ----------
    clients_data_info: dict[str, dict[str, any]]
        Client-wise data-info dict that are to be aggregated.
    required_fields: set[str]
        Optional set of fields to target among provided information.
        If set, raise if a field is missing from any client, and use
        only these fields in the returned dict.

    Raises
    ------
    AggregationError:
        In case any error is raised when calling `aggregate_data_info`
        on the input arguments.

    Returns
    -------
    data_info: dict[str, any]
        Aggregated data specifications derived from individual ones,
        with `required_fields` as keys.
    """
    try:
        return aggregate_data_info(
            list(clients_data_info.values()), required_fields
        )
    except KeyError as exc:
        _raise_on_missing_fields(clients_data_info, required_fields)
        _raise_aggregation_fails(clients_data_info, exc)
    except ValueError as exc:
        _raise_on_invalid_fields(clients_data_info, required_fields)
        _raise_incompatible_fields(clients_data_info, exc)
    except Exception as exc:  # re-raise; pylint: disable=broad-except
        _raise_aggregation_fails(clients_data_info, exc)


def _raise_on_missing_fields(
    clients_data_info: Dict[str, Dict[str, Any]],
    required_fields: Set[str],
) -> None:
    """Raise information about missing-fields-due data_info agg. failure.

    Raise a RuntimeError containing client-wise messages and server error.
    Return None if no missing field was encountered.
    """
    # Identify missing fields.
    errors = {}  # type: Dict[str, str]
    for client, data_info in clients_data_info.items():
        missing = required_fields.difference(data_info.keys())
        if missing:
            errors[client] = f"Missing 'data_info' fields: {missing}"
    # If no errors are encountered, return none.
    if not errors:
        return None
    # Prepare messages to send to clients.
    default = "Other clients had missing 'data_info' fields."
    messages = {
        client: errors.get(client, default) for client in clients_data_info
    }
    # Prepare error message to log and raise by the server.
    error = "Some clients submitted incomplete 'data_info':\n"
    error += "\n".join(
        f"{client}: {message}" for client, message in errors.items()
    )
    # Raise, including both.
    raise AggregationError(error, messages)


def _raise_aggregation_fails(
    clients_data_info: Dict[str, Dict[str, Any]],
    exception: Exception,
) -> NoReturn:
    """Raise information about aggregation failure for unexpected cause.

    Raise a RuntimeError containing client-wise messages and server error.
    """
    # Prepare error message to log and raise by the server.
    error = "Unexpected error while aggregating 'data_info':\n"
    error += f"{type(exception).__name__} {exception}"
    # Prepare messages to send to clients.
    messages = {client: error for client in clients_data_info}
    # Raise, including both.
    raise AggregationError(error, messages)


def _raise_on_invalid_fields(
    clients_data_info: Dict[str, Dict[str, Any]],
    required_fields: Set[str],
) -> None:
    """Raise information about invidual-values-due data_info agg. failure.

    Raise a RuntimeError containing client-wise messages and server error.
    Return None if no client-due value error was encountered.
    """
    # Identify missing fields.
    errors = {}  # type: Dict[str, str]
    for client, data_info in clients_data_info.items():
        try:
            aggregate_data_info([data_info], required_fields)
        except ValueError as exc:
            errors[client] = f"Invalid 'data_info': ValueError {exc}"
    # If no errors are encountered, return none.
    if not errors:
        return None
    # Prepare messages to send to clients.
    default = "Other clients had invalid 'data_info' values."
    messages = {
        client: errors.get(client, default) for client in clients_data_info
    }
    # Prepare error message to log and raise by the server.
    error = "Some clients submitted invalid 'data_info' values:\n"
    error += "\n".join(
        f"{client}: {message}" for client, message in errors.items()
    )
    # Raise, including both.
    raise AggregationError(error, messages)


def _raise_incompatible_fields(
    clients_data_info: Dict[str, Dict[str, Any]],
    exception: ValueError,
) -> NoReturn:
    """Raise information about incompatible-due data_info agg. failure.

    Raise a RuntimeError containing client-wise messages and server error.
    """
    # Prepare messages to send to clients.
    default = f"Incompatible 'data_info': ValueError {exception}"
    messages = {client: default for client in clients_data_info}
    # Prepare error message to log and raise by the server.
    error = "Incompatibility between submitted 'data_info' values:\n"
    error += f"ValueError {exception}"
    # Raise, including both.
    raise AggregationError(error, messages)
