# coding: utf-8

"""Strategy scratch code rewrite."""

from abc import ABCMeta, abstractmethod
import dataclasses
from typing import Any, Dict, List, Union


from declearn.optimizer import Optimizer
from declearn.optimizer.modules import OptiModule
from declearn.strategy._aggregator import Aggregator, AverageAggregator
from declearn.utils import deserialize_object, json_load


__all__ = [
    "FedAvg",
    "Strategy",
    "strategy_from_config",
]


class Strategy(metaclass=ABCMeta):
    """Base class to define a client/server FL Strategy.

    This class is meant to design an API enabling the modular design
    of Federated Learning strategies, which are defined by:
    * an updates-aggregation algorithm
    * a server-side optimization algorithm, to refine and apply
      aggregated updates
    * a client-side optimization algorithm, to refine and apply
      step-wise gradient-based updates
    * (opt.) a client-sampling policy, to select participating
      clients to a given training round

    At the moment, the design of this class is *unfinished*.
    Notably, in addition to the algorithmic modularity, the
    future aim will be to have a modular way to instantiate
    a strategy (e.g. using configuration files, authorizing
    some level of client-wise overload, etc.).
    """

    @abstractmethod
    def build_server_aggregator(
        self,
    ) -> Aggregator:
        """Set up and return an Aggregator to be used by the server."""
        raise NotImplementedError

    @abstractmethod
    def build_server_optimizer(
        self,
    ) -> Optimizer:
        """Set up and return an Optimizer to be used by the server."""
        raise NotImplementedError

    def _build_server_modules(
        self,
    ) -> List[OptiModule]:
        """Return a list of OptiModule plug-ins for the server to use."""
        return []

    @abstractmethod
    def build_client_optimizer(
        self,
    ) -> Optimizer:
        """Set up and return an Optimizer to be used by clients."""
        raise NotImplementedError

    def _build_client_modules(
        self,
    ) -> List[OptiModule]:
        """Return a list of OptiModule plug-ins for clients to use."""
        return []

    # revise: add this once clients-sampling is implemented
    # @abstractmethod
    # def build_clients_sampler(
    #         self,
    #     ) -> ClientsSelector:
    #     """Docstring."""


@dataclasses.dataclass
class AggregConfig:
    """Dataclass specifying server aggregator config (and default)."""

    name: str = "Average"
    group: str = "Aggregator"
    config: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ClientConfig:
    """Dataclass specifying client-side optimizer config (and default)."""

    lrate: float = 1e-4
    w_decay: float = 0.0
    modules: List[OptiModule] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ServerConfig:
    """Dataclass specifying server-side optimizer config (and default)."""

    lrate: float = 1.0
    w_decay: float = 0.0
    modules: List[OptiModule] = dataclasses.field(default_factory=list)


def strategy_from_config(  # revise: generalize this (into Strategy?)
    config: Union[str, Dict[str, Any]],
) -> Strategy:
    """Define a custom Strategy from a configuration file."""
    if isinstance(config, str):
        config = json_load(config)
    if not isinstance(config, dict):
        raise TypeError("'config' should be a dict or JSON-file-stored dict.")
    # Parse the configuration dict (raise if keys are unproper).
    aggreg_cfg = AggregConfig(**config.get("aggregator", {}))
    client_cfg = ClientConfig(**config.get("client_opt", {}))
    server_cfg = ServerConfig(**config.get("server_opt", {}))
    # Declare a custom class that makes use of the previous.
    class CustomStrategy(Strategy):
        """Custom strategy defined from a configuration file."""

        def build_server_aggregator(self) -> Aggregator:
            cfg = dataclasses.asdict(aggreg_cfg)
            agg = deserialize_object(cfg)  # type: ignore
            if not isinstance(agg, Aggregator):
                raise TypeError("Unproper object instantiated as aggregator.")
            return agg

        def build_server_optimizer(self) -> Optimizer:
            return Optimizer(**dataclasses.asdict(server_cfg))

        def build_client_optimizer(self) -> Optimizer:
            return Optimizer(**dataclasses.asdict(client_cfg))

    # Instantiate from the former and return.
    return CustomStrategy()


class FedAvg(Strategy):
    """FedAvg Strategy defining class.

    FedAvg is one of the simplest Federated Learning strategies
    existing. This implementation allows for a few tricks to be
    used, but has default values that leave these out.

    FedAvg is characterized by:
    * A simple averaging of local updates to aggregate them into
      global updates (here with a default behaviour to reweight
      clients' contributions based on the number of steps taken).
    * The use of simple SGD by clients (here with a default step
      side of 1e-4 and optional weight decay).
    * The absence of refinement of averaged updates on the server
      side (here with the possibility to enforce a learning rate,
      aka a slowdown parameter, and use optional weight decay).
    """

    def __init__(
        self,
        eta_l: float = 1e-4,
        eta_g: float = 1.0,
        lam_l: float = 0.0,
        lam_g: float = 0.0,
    ) -> None:
        """Instantiate the FedAvg Strategy.

        Parameters
        ----------
        eta_l: float, default=0.0001,
            Learning rate parameter of clients' optimizer.
        eta_g: float, default=1.
            Learning rate parameter of the server's optimizer.
            Defaults to 1 so as to merely average local updates.
        lam_l: float, default=0.
            Weight decay parameter of clients' optimizer.
            Defaults to 0 so as not to use any weight decay.
        lam_g: float, default=0.
            Weight decay parameter of the server's optimizer.
            Defaults to 0 so as not to use any weight decay.
        """
        self.eta_l = eta_l
        self.eta_g = eta_g
        self.lam_l = lam_l
        self.lam_g = lam_g

    def build_client_optimizer(
        self,
    ) -> Optimizer:
        modules = self._build_client_modules()
        return Optimizer(self.eta_l, self.lam_l, modules=modules)

    def build_server_optimizer(
        self,
    ) -> Optimizer:
        modules = self._build_server_modules()
        return Optimizer(self.eta_g, self.lam_g, modules=modules)

    def build_server_aggregator(
        self,
    ) -> Aggregator:
        return AverageAggregator(steps_weighted=True)
