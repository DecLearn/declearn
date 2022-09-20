# coding: utf-8

"""Strategy scratch code rewrite."""

from abc import ABCMeta
from typing import List


from declearn.optimizer.modules import (
    MomentumModule, OptiModule, ScaffoldClientModule, ScaffoldServerModule
)
from declearn.strategy._strategy import FedAvg, Strategy


__all__ = [
    'FedAvgM',
    'Scaffold',
    'ScaffoldM',
]


class FedAvgM(FedAvg):
    """FedAvgM Strategy defining class.

    FedAvgM, or FedAvg with Momentum, is a Strategy extending
    FedAvg to use momentum when applying aggregated updates
    on the server side.
    """

    def __init__(
            self,
            eta_l: float = 1e-4,
            eta_g: float = 1.,
            lam_l: float = 0.,
            lam_g: float = 0.,
            beta: float = 0.9,
        ) -> None:
        """Instantiate the FedAvgM Strategy.

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
        beta: float, default=.9
            Momentum parameter applied to aggregated updates.
            See `declearn.optimizer.modules.MomentumModule`.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        super().__init__(eta_l, eta_g, lam_l, lam_g)
        self.beta = beta

    def _build_server_modules(
            self,
        ) -> List[OptiModule]:
        modules = super()._build_server_modules()
        modules.append(MomentumModule(self.beta))
        return modules


class _ScaffoldMixin(Strategy, metaclass=ABCMeta):
    """Mix-in class to use SCAFFOLD on top of a base Strategy.

    SCAFFOLD, or Stochastic Controlled Averaging for Federated
    Learning, is a modification of the base federated learning
    process to regularize clients' drift away from the shared
    model during the local training steps.

    It is implemented using a pair of OptiModule objects that
    exchange and maintain state variables through training.

    See `declearn.optimizer.modules.ScaffoldClientModule` and
    `ScaffoldServerModule` for details.
    """

    def _build_server_modules(
            self,
        ) -> List[OptiModule]:
        modules = super()._build_server_modules()
        modules.append(ScaffoldServerModule())
        return modules

    def _build_client_modules(
            self,
        ) -> List[OptiModule]:
        modules = super()._build_client_modules()
        modules.append(ScaffoldClientModule())
        return modules


class Scaffold(_ScaffoldMixin, FedAvg):
    """Scaffold Strategy defining class.

    SCAFFOLD, or Stochastic Controlled Averaging for Federated Learning,
    is a modification of FedAvg that applies a correction term to local
    gradients at each SGD step in order to prevent clients' models from
    drifting away too much from the shared model.

    It relies on the use of state variables that are maintained through
    time and updated between rounds, based on the clients' sharing state
    information with the server and receiving updates in return.

    This class implements SCAFFOLD on top of the base FedAvg Strategy.
    See `declearn.optimizer.modules.ScaffoldClientModule` and
    `ScaffoldServerModule` for details on SCAFFOLD.
    See `declearn.strategy.FedAvg` for the base FedAvg class.
    """


class ScaffoldM(_ScaffoldMixin, FedAvgM):
    """ScaffoldM Strategy defining class.

    ScaffoldM is SCAFFOLD (see `declearn.strategy.Scaffold`) combined
    with the use of momentum when applying aggregated upgrades to the
    global model. In other words, it is SCAFFOLD on top of FedAvgM.
    """
