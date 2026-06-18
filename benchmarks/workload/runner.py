"""Run a `BenchmarkSpec` end-to-end as a local federated experiment."""

import asyncio
import os

from declearn.communication import NetworkClientConfig, NetworkServerConfig
from declearn.main import FederatedClient, FederatedServer
from declearn.utils import set_device_policy

from benchmarks.workload.spec import BenchmarkSpec

__all__ = ["run_benchmark"]


_FORCE_GPU_ENV = "DECLEARN_BENCH_FORCE_GPU"


def _enforce_device_policy() -> None:
    """If `DECLEARN_BENCH_FORCE_GPU=1`, refuse to run on CPU.

    Why: a misconfigured cluster env (wrong CUDA toolchain, missing
    driver, etc.) will silently fall back to CPU and produce timings
    that look like a 10x regression. In CI / cluster runs, set
    `DECLEARN_BENCH_FORCE_GPU=1` to fail loudly instead.
    """
    if os.environ.get(_FORCE_GPU_ENV, "").strip() not in ("1", "true", "yes"):
        return
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"{_FORCE_GPU_ENV} is set but PyTorch is not installed."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{_FORCE_GPU_ENV} is set but `torch.cuda.is_available()` is "
            "False. Refusing to run on CPU. Check the CUDA driver and "
            "the PyTorch CUDA build."
        )
    set_device_policy(gpu=True)


async def _run_async(spec: BenchmarkSpec) -> None:
    """Spawn the server and clients on the event loop and await completion."""
    server_uri = (
        f"ws://{spec.network_host}:{spec.network_port}"
        if spec.network_protocol == "websockets"
        else f"{spec.network_protocol}://"
        f"{spec.network_host}:{spec.network_port}"
    )
    netwk_server = NetworkServerConfig.from_params(
        protocol=spec.network_protocol,
        host=spec.network_host,
        port=spec.network_port,
        heartbeat=0.1,
    )
    server = FederatedServer(
        model=spec.server_model,
        netwk=netwk_server,
        optim=spec.optim_config,
        metrics=spec.metrics or None,
        secagg=spec.server_secagg,
    )

    client_coros = []
    for client_spec in spec.clients:
        netwk_client = NetworkClientConfig.from_params(
            protocol=spec.network_protocol,
            server_uri=server_uri,
            name=client_spec.name,
        )
        client = FederatedClient(
            netwk=netwk_client,
            train_data=client_spec.train_data,
            valid_data=client_spec.valid_data,
            secagg=client_spec.secagg,
            share_metrics=True,
            verbose=False,
        )
        client_coros.append(client.async_run())

    await asyncio.gather(server.async_run(spec.run_config), *client_coros)


def run_benchmark(spec: BenchmarkSpec) -> None:
    """Synchronous entry point: run the benchmark to completion."""
    _enforce_device_policy()
    asyncio.run(_run_async(spec))
