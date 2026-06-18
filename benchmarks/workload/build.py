"""Translate benchmark parameters into a fully-instantiated `BenchmarkSpec`.

This is the layer where parameter interpretation lives: a small set of
high-level toggles ("torch + SCAFFOLD + 5 clients") expand into concrete
declearn objects (Model, FLOptimConfig, FLRunConfig, datasets, optional
SecAgg configs). Validation rejects parameter combinations that declearn
or the optional dependencies cannot honor (e.g. SCAFFOLD on non-torch
backends in the v1 suite).
"""

import importlib
from typing import List, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from declearn.dataset import InMemoryDataset
from declearn.main.config import FLOptimConfig, FLRunConfig
from declearn.model.api import Model
from declearn.secagg.api import SecaggConfigClient, SecaggConfigServer
from declearn.secagg.utils import IdentityKeys

from benchmarks.workload import baseline as B
from benchmarks.workload.data import (
    client_data_paths,
    ensure_data_for_n_clients,
)
from benchmarks.workload.spec import BenchmarkSpec, ClientSpec

__all__ = ["BACKEND_LAYOUT", "build_benchmark"]


_VALID_BACKENDS = ("torch", "tensorflow")
_VALID_REGULARIZERS = (None, "ridge", "fedprox")
_VALID_SECAGG = (None, "masking")

BACKEND_LAYOUT = {
    "torch": "chw",
    "tensorflow": "hwc",
}

_BACKEND_MODEL_MODULE = {
    "torch": "benchmarks.workload.models.torch_cnn",
    "tensorflow": "benchmarks.workload.models.tensorflow_cnn",
}


def _validate(
    backend: str,
    regularizer: Optional[str],
    scaffold: bool,
    secagg: Optional[str],
) -> None:
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Invalid backend '{backend}'. Expected one of {_VALID_BACKENDS}."
        )
    if regularizer not in _VALID_REGULARIZERS:
        raise ValueError(
            f"Invalid regularizer '{regularizer}'. "
            f"Expected one of {_VALID_REGULARIZERS}."
        )
    if secagg not in _VALID_SECAGG:
        raise ValueError(
            f"Invalid secagg '{secagg}'. Expected one of {_VALID_SECAGG}."
        )
    if scaffold and backend != "torch":
        raise ValueError(
            "SCAFFOLD is restricted to backend='torch' in the v1 "
            "benchmark suite."
        )


def _build_model(backend: str) -> Model:
    module = importlib.import_module(_BACKEND_MODEL_MODULE[backend])
    return module.build_model()


def _build_optim(
    regularizer: Optional[str], scaffold: bool
) -> FLOptimConfig:
    client_modules: List = list(B.BASELINE_CLIENT_MODULES)
    if scaffold:
        client_modules.append("scaffold-client")
    client_opt = {
        "lrate": B.BASELINE_CLIENT_LRATE,
        "modules": client_modules,
    }
    if regularizer is not None:
        client_opt["regularizers"] = [regularizer]
    server_modules: List = []
    if scaffold:
        server_modules.append("scaffold-server")
    server_opt = {
        "lrate": B.BASELINE_SERVER_LRATE,
        "modules": server_modules or None,
    }
    return FLOptimConfig.from_params(
        aggregator=B.BASELINE_AGGREGATOR,
        client_opt=client_opt,
        server_opt=server_opt,
    )


def _build_run_config(
    rounds: int,
    n_clients: int,
    batch_size: int,
) -> FLRunConfig:
    return FLRunConfig.from_params(
        rounds=rounds,
        register={
            "min_clients": n_clients,
            "timeout": B.BASELINE_REGISTRATION_TIMEOUT,
        },
        training={"batch_size": batch_size},
        evaluate={"batch_size": B.BASELINE_EVAL_BATCH_SIZE},
    )


def _build_secagg(
    secagg: Optional[str], n_clients: int
) -> tuple[Optional[SecaggConfigServer], List[Optional[SecaggConfigClient]]]:
    if secagg is None:
        return None, [None] * n_clients
    # _validate guarantees secagg == "masking" here.
    from declearn.secagg.masking import (  # noqa: PLC0415
        MaskingSecaggConfigClient,
        MaskingSecaggConfigServer,
    )

    private_keys = [Ed25519PrivateKey.generate() for _ in range(n_clients)]
    public_keys = [key.public_key() for key in private_keys]
    id_keys = [
        IdentityKeys(prv, trusted=public_keys) for prv in private_keys
    ]
    server_cfg: SecaggConfigServer = MaskingSecaggConfigServer(
        bitsize=64, clipval=1e8
    )
    client_cfgs: List[Optional[SecaggConfigClient]] = [
        MaskingSecaggConfigClient(id_keys=keys) for keys in id_keys
    ]
    return server_cfg, client_cfgs


def _build_clients(
    n_clients: int,
    layout: str,
    client_secagg: List[Optional[SecaggConfigClient]],
) -> List[ClientSpec]:
    ensure_data_for_n_clients(n_clients, layout)
    clients: List[ClientSpec] = []
    for idx in range(n_clients):
        train_d, train_t, valid_d, valid_t = client_data_paths(
            n_clients, layout, idx
        )
        train = InMemoryDataset(
            data=str(train_d),
            target=str(train_t),
            expose_classes=True,
        )
        valid = InMemoryDataset(
            data=str(valid_d),
            target=str(valid_t),
        )
        clients.append(
            ClientSpec(
                name=f"client_{idx}",
                train_data=train,
                valid_data=valid,
                secagg=client_secagg[idx],
            )
        )
    return clients


def build_benchmark(  # noqa: PLR0913 — flat axis API by design
    backend: str = B.BASELINE_BACKEND,
    n_clients: int = B.BASELINE_N_CLIENTS,
    regularizer: Optional[str] = None,
    scaffold: bool = False,
    secagg: Optional[str] = None,
    rounds: int = B.BASELINE_ROUNDS,
    batch_size: int = B.BASELINE_BATCH_SIZE,
) -> BenchmarkSpec:
    """Assemble a fully-instantiated `BenchmarkSpec`.

    Parameters
    ----------
    backend:
        One of `"torch"`, `"tensorflow"`.
    n_clients:
        Number of federated clients to spawn.
    regularizer:
        Optional client-side loss regularizer. One of `None`, `"ridge"`,
        `"fedprox"`.
    scaffold:
        Whether to enable SCAFFOLD aux-var exchange. Requires
        `backend="torch"` in the v1 suite.
    secagg:
        Optional secure-aggregation method. One of `None`, `"masking"`.
    rounds:
        Number of federated training rounds.
    batch_size:
        Per-client training batch size.

    Raises
    ------
    ValueError
        On invalid parameter combinations (e.g. SCAFFOLD on a non-torch
        backend in the v1 suite).
    """
    _validate(backend, regularizer, scaffold, secagg)
    layout = BACKEND_LAYOUT[backend]
    server_model = _build_model(backend)
    optim = _build_optim(regularizer, scaffold)
    run = _build_run_config(rounds, n_clients, batch_size)
    server_secagg, client_secagg = _build_secagg(secagg, n_clients)
    clients = _build_clients(n_clients, layout, client_secagg)
    return BenchmarkSpec(
        server_model=server_model,
        optim_config=optim,
        run_config=run,
        network_host=B.BASELINE_NETWORK_HOST,
        network_port=B.BASELINE_NETWORK_PORT,
        network_protocol=B.BASELINE_NETWORK_PROTOCOL,
        clients=clients,
        server_secagg=server_secagg,
        metrics=list(B.BASELINE_METRICS),
    )
