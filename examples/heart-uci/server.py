"""Script to run a federated server on the heart-disease example."""

import argparse
import os

from declearn.communication import NetworkServerConfig
from declearn.main import FederatedServer
from declearn.model.sklearn import SklearnSGDModel
from declearn.optimizer.modules import EWMAModule, RMSPropModule
from declearn.strategy import strategy_from_config

FILEDIR = os.path.dirname(os.path.abspath(__file__))


def run_server(
    nb_clients: int,
    sv_cert: str,
    sv_priv: str,
) -> None:
    """Instantiate and run the orchestrating server.

    Arguments
    ---------
    nb_clients: int
        Exact number of clients used in this example.
    sv_cert: str
        Path to the (self-signed) SSL certificate to use.
    sv_priv: str
        Path to the associated private-key to use.
    """

    # (1) Define a model

    # Here we use a scikit-learn SGD classifier and parametrize it
    # into a L2-penalized binary logistic regression.
    model = SklearnSGDModel.from_parameters(
        kind="classifier", loss="log_loss", penalty="l2", alpha=0.005
    )

    # (2) Define a strategy

    # Configure the aggregator to use.
    # Here, averaging weighted by the effective number
    # of local gradient descent steps taken.
    aggregator = {
        "name": "Average",
        "config": {"steps_weighted": True},
    }

    # Configure the client-side optimizer to use.
    # Here, RMSProp optimizer with 0.02 learning rate.
    client_opt = {
        "lrate": 0.02,
        "modules": [RMSPropModule()],
    }

    # Configure the server-side optimizer to use.
    # Here, apply momentum to the updates and apply them (as lr=1.0).
    server_opt = {
        "lrate": 1.0,
        "modules": [EWMAModule()],
    }

    # Wrap this up into a Strategy object$
    config = {
        "aggregator": aggregator,
        "client_opt": client_opt,
        "server_opt": server_opt,
    }
    strategy = strategy_from_config(config)

    # (3) Define network communication parameters.

    # Here, use websockets protocol on localhost:8765, with SSL encryption.
    network = NetworkServerConfig(
        protocol="websockets",
        host="localhost",
        port=8765,
        certificate=sv_cert,
        private_key=sv_priv,
    )

    # (4) Instantiate and run a FederatedServer.

    server = FederatedServer(model, network, strategy)
    # Here, we setup 20 rounds of training, with 30 samples per batch
    # during training and 50 during validation; plus an early-stopping
    # criterion if the global validation loss stops decreasing for 5 rounds.
    server.run(
        rounds=20,
        regst_cfg={"min_clients": nb_clients},
        train_cfg={"batch_size": 30, "drop_remainder": False},
        valid_cfg={"batch_size": 50, "drop_remainder": False},
        early_stop={"tolerance": 0.0, "patience": 5, "relative": False},
    )


# Called when the script is called directly (using `python server.py`).
if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nb_clients",
        type=int,
        help="number of clients",
        choices=[1, 2, 3, 4],
    )
    parser.add_argument(
        "--cert_path",
        dest="cert_path",
        type=str,
        help="path to the server-side ssl certification",
        default=os.path.join(FILEDIR, "server-cert.pem"),
    )
    parser.add_argument(
        "--key_path",
        dest="key_path",
        type=str,
        help="path to the server-side ssl private key",
        default=os.path.join(FILEDIR, "server-key.pem"),
    )
    args = parser.parse_args()
    # Run the server routine.
    run_server(args.nb_clients, args.cert_path, args.key_path)
