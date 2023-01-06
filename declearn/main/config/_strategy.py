# coding: utf-8

"""TOML-parsable container for a Federated Learning optimization strategy."""

import dataclasses
import functools
from typing import Any, Dict, Union


from declearn.aggregator import Aggregator
from declearn.optimizer import Optimizer
from declearn.utils import TomlConfig, access_registered, deserialize_object


__all__ = [
    "FLOptimConfig",
]


@dataclasses.dataclass
class FLOptimConfig(TomlConfig):
    """Container dataclass for a federated optimization strategy.

    This dataclass is designed to wrap together an Aggregator and
    a pair of Optimizer instances, that are respectively meant to
    be used by the server and the clients. The main point of this
    class is to provide with TOML-parsing capabilities, so that a
    strategy can be specified via a TOML file, which is expected
    to be simpler to edit and maintain than direct Python code by
    end-users.

    It is designed to be used by the orchestrating server in the
    case of a centralized federated learning process.

    Fields
    ------
    client_opt: Optimizer
        Optimizer to be used by clients (that each hold a copy)
        so as to conduct the step-wise local model updates.
    server_opt: Optimizer, default=Optimizer(lrate=1.0)
        Optimizer to be used by the server so as to conduct a
        round-wise global model update based on the aggregated
        client updates.
    aggregator: Aggregator, default=AverageAggregator()
        Client weights aggregator to be used by the server so as
        to conduct the round-wise aggregation of client udpates.

    Notes
    -----
    The `aggregator` field may be specified in a variety of ways:
    - a single string may specify the registered name of the class
      constructor to use.
      In TOML, use `aggregator = "<name>"` outside of any section.
    - a serialization dict, that specifies the registration `name`,
      and optionally a registration `group` and/or arguments to be
      passed to the class constructor.
      In TOML, use an `[aggregator]` section with a `name = "<name>"`
      field and any other fields you wish to pass. Kwargs may either
      be grouped into a dedicated `[aggregator.config]` sub-section
      or provided as fields of the main aggregator section.

    Instantiation classmethods
    --------------------------
    from_toml:
        Instantiate by parsing a TOML configuration file.
    from_params:
        Instantiate by parsing inputs dicts (or objects).
    """

    client_opt: Optimizer
    server_opt: Optimizer = dataclasses.field(
        default_factory=functools.partial(Optimizer, lrate=1.0)
    )
    aggregator: Aggregator = dataclasses.field(default_factory=Aggregator)

    @classmethod
    def parse_aggregator(
        cls,
        field: dataclasses.Field,  # future: dataclasses.Field[Aggregator]
        inputs: Union[str, Dict[str, Any], Aggregator],
    ) -> Aggregator:
        """Field-specific parser to instantiate an Aggregator.

        This method supports specifying `aggregator`:
        * as a str, used to retrieve a registered Aggregator class
        * as a dict, parsed a serialized Aggregator configuration:
            - name: str used to retrieve a registered Aggregator class
            - (opt.) group: str used to retrieve the registered class
            - (opt.) config: dict specifying kwargs for the constructor
            - any other field will be added to the `config` kwargs dict
        * as None (or missing kwarg), using default AverageAggregator()
        """
        # Case when using the default value: delegate to the default parser.
        if inputs is None:
            return cls.default_parser(field, inputs)
        # Case when the input is a valid instance: return it.
        if isinstance(inputs, Aggregator):
            return inputs
        # Case when provided with a string: retrieve the class and instantiate.
        if isinstance(inputs, str):
            try:
                # Note: subclass-checking is performed by `access_registered`.
                agg_cls = access_registered(inputs, group="Aggregator")
            except KeyError as exc:
                raise TypeError(
                    f"Failed to retrieve Aggregator class from name '{inputs}'"
                ) from exc
            return agg_cls()
        # Case when provided with a dict: check/fix formatting and deserialize.
        if isinstance(inputs, dict):
            if "name" not in inputs:
                raise TypeError(
                    "Wrong format for Aggregator serialized config: missing "
                    "'name' field."
                )
            inputs.setdefault("group", "Aggregator")
            inputs.setdefault("config", {})
            for key in list(inputs):
                if key not in ("name", "group", "config"):
                    inputs["config"][key] = inputs.pop(key)
            obj = deserialize_object(inputs)  # type: ignore
            if not isinstance(obj, Aggregator):
                raise TypeError(
                    "Input specifications for 'aggregator' resulted in a non-"
                    f"Aggregator object with type '{type(obj)}'."
                )
            return obj
        # Otherwise, raise a TypeError as inputs are unsupported.
        raise TypeError("Unsupported inputs type for field 'aggregator'.")
