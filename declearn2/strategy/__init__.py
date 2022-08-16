# coding: utf-8

"""Federated Learning Strategy definition API and examples submodule."""

from ._aggregator import (
    Aggregator,
    AverageAggregator,
)
from ._strategy import (
    FedAvg,
    Strategy,
    strategy_from_config,
)
from ._strategies import (
    FedAvgM,
    Scaffold,
    ScaffoldM,
)
