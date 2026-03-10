# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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

"""Demonstration script using the MNIST dataset on a Grid5000 deployment."""

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import enoslib as en
import fire

# CONSTANTS (you can change their value to modify the experiment setup)
DATA_DIR = "mnist_data"
"""
Name of the directory (on the network storage) in which the data will be
downloaded.
"""

DATA_SCHEME = "iid"
"""Data splitting scheme to use, see prepare_data.py for details."""

WALLTIME = "01:59:00"
"""Maximum real execution time reserved to run an experiment on Grid5000."""

SERVER_CLUSTER = "nova"
"""
Name of the cluster on which the server machine will be reserved.
For the list of valid cluster names, see: https://www.grid5000.fr/w/Hardware
"""

CLUSTER_TO_NB_CLIENTS: Dict[str, int] = {
    "dahu": 1,
    "econome": 2,
}
"""
Dictionary mapping the name of a machine cluster to the number of clients 
we want to deploy on it (each client is on a distinct machine of the cluster).
"""

# EnOSlib init
en.init_logging(level=logging.INFO)
en.check()  # check and print the status of EnOSlib.


def run_grid5k(
    storage_path: str,
) -> None:
    """Main script: Run a federated learning experiment using ENOSLIB.

    Parameters
    ------
    storage_path: str
        Path to your network storage (ex: group storage), accessible from
        Grid5000 machines.
        E.g. `/srv/storage/my_storage_name@storage1.lille.grid5000.fr/myuser/
        declearn-grid5k`
    """
    roles, _, provider = get_resources()
    server = roles["server"][0]
    nb_clients = len(roles["client"])
    host = server.address

    config_resources(
        nb_clients=nb_clients,
        storage_path=storage_path,
        data_dir=DATA_DIR,
        roles=roles,
        host=host,
    )
    print(f"Starting experiment...")
    run_server(
        nb_clients=nb_clients,
        roles=roles,
        host=host,
    )
    run_clients(
        nb_clients=nb_clients,
        storage_path=storage_path,
        data_dir=DATA_DIR,
        roles=roles,
        host=host,
    )
    # Wait for background processes and pull results out.
    time.sleep(60)
    en.run_command(
        f"mkdir -v -p {os.path.join(storage_path, 'results')}",
        roles=roles["server"],
    )
    en.run_command(
        f"cp -r result* {os.path.join(storage_path, 'results/')}",
        roles=roles,
        on_error_continue=True,
    )
    run_cmd(
        f"sudo chmod --recursive 777 {os.path.join(storage_path, 'results')}",
        roles=roles["server"],
    )
    # Release resources.
    provider.destroy()


def get_resources() -> Tuple[
    en.objects.Roles, en.objects.Networks, en.infra.enos_g5k.provider.G5k
]:
    """Connect to Grid5000 and get computing resources.

    Returns
    -------
    roles: en.objects.Roles
        enoslib dict for HostsView.
    networks: en.objects.Networks
        enoslib dict for NetworksView.
    provider:
        G5k object for interacting with Grid 5000.
    """
    job_name = Path(__file__).name + datetime.now().strftime("%d%H%M%S")
    h, m, s = map(int, WALLTIME.split(":"))
    walltime_delta = timedelta(hours=h, minutes=m, seconds=s)
    getting_resources = True
    while getting_resources:
        now = datetime.now()
        # Day regime is between 9:00 and 19:00.
        day_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        day_end = now.replace(hour=19, minute=0, second=0, microsecond=0)

        if (
            now.weekday() < 5
            and day_start <= now < day_end
            and day_start <= now + walltime_delta <= day_end
        ):
            job_type = ["day", "deploy"]
        else:
            job_type = ["night", "deploy"]

        conf = en.G5kConf.from_settings(
            job_name=job_name,
            job_type=job_type,
            env_name="ubuntu2204-nfs",
            walltime=WALLTIME,
        ).add_machine(roles=["server"], cluster=SERVER_CLUSTER, nodes=1)

        # Add client machines to the configuration.
        for cluster, nb_clients in CLUSTER_TO_NB_CLIENTS.items():
            conf = conf.add_machine(
                roles=["client"], cluster=cluster, nodes=nb_clients
            )

        try:
            # Validate the configuration.
            provider = en.G5k(conf)
            # Get actual resources.
            roles, networks = provider.init()
            getting_resources = False
        except Exception as error:
            print("Error getting g5k resources:", error)
            provider.destroy()
            getting_resources = True
            print("Unable to get g5k resources. Retrying in half an hour...")
            time.sleep(30 * 60)
    return roles, networks, provider


def config_resources(
    nb_clients: int,
    storage_path: str,
    data_dir: str,
    roles: list[en.objects.Roles],
    host: str,
) -> None:
    """Configure computing and data resources to deploy training.

    Parameters
    ------
    nb_clients: int
        Number of clients involved in the experiments.
    storage_path: str
        Path to your network storage (ex: group storage), accessible from
        Grid5000 machines.
    data_dir: str
        Name of the directory (on the network storage) in which the data will be
        downloaded.
    roles: list[en.objects.Roles]
        Instances on which the command line will be executed.
    host: str
        Address of the instance running the server.
    """
    # Setup environment.
    run_cmd("export DEBIAN_FRONTEND=noninteractive", roles)
    run_cmd("sudo -E add-apt-repository -y ppa:deadsnakes/ppa", roles)
    run_cmd("sudo -E apt-get update -y", roles)
    run_cmd("sudo -E apt-get install -y git", roles)
    run_cmd("sudo -E apt-get install -y software-properties-common", roles)
    run_cmd(
        "sudo -E apt-get install -y python3.11 python3.11-venv",
        roles,
    )

    # Create declearn venv in "/opt/dln-venv".
    run_cmd("python3.11 -m venv /opt/dln-venv", roles)
    python_bin = "/opt/dln-venv/bin/python"  # short alias for venv's python
    run_cmd(
        f"{python_bin} -m pip install --upgrade pip setuptools wheel", roles
    )
    run_cmd(
        f"git clone --branch develop --single-branch --depth 1 "
        "https://gitlab.inria.fr/magnet/declearn/declearn2.git /opt/declearn",
        roles,
    )
    run_cmd(f"{python_bin} -m pip install -e /opt/declearn[all]", roles)

    # Copy .py scripts from mnist example folder to the current path
    # for simplicity purpose :
    run_cmd("cp -v /opt/declearn/examples/mnist_g5k/*py ./", roles)

    # Generate SSL certificates and artifacts on the server machine.
    run_cmd(
        f"{python_bin} generate_ssl.py --c_name {host}",
        roles["server"],
    )
    ## Copy public certificate on the shared network storage.
    run_cmd(
        f"cp ./ca-cert.pem {os.path.join(storage_path, 'ca-cert.pem')}",
        roles["server"],
    )

    # Prepare data on the shared network storage.
    data_path = os.path.join(storage_path, data_dir)
    run_cmd(
        f"mkdir -v -p {data_path}",
        roles["server"],
    )
    run_cmd(
        f"{python_bin} prepare_data.py {nb_clients} --scheme {DATA_SCHEME} \
            --folder {data_path}",
        roles["server"],
    )


def run_cmd(
    cmd: str,
    roles: list[en.objects.Roles],
) -> None:
    """
    Use EnOSlib to run the command line on provided instances,
    printing stdout.

    Parameters
    ------
    cmd: str
        command line to run.
    roles: list[en.objects.Roles]
        Intances where the command line will be executed.
    """
    results = en.run_command(cmd, roles=roles)
    for result in results:
        print(result.payload["stdout"])


def run_server(
    nb_clients: int,
    roles: list[en.objects.Roles],
    host: str,
) -> None:
    """Run server in a Grid5000 instance.

    Parameters
    ------
    nb_clients: int
        Number of clients involved in the experiments.
    roles: list[en.objects.Roles]
        Instances where the command line will be executed.
    host: str
        Address of the instance running the server.
    """
    python_bin = "/opt/dln-venv/bin/python"
    en.run_command(
        f"{python_bin} run_server.py --nb_clients {nb_clients} --host {host}",
        background=True,
        roles=roles["server"],
    )


def run_clients(
    nb_clients: int,
    storage_path: str,
    data_dir: str,
    roles: list[en.objects.Roles],
    host: str,
) -> None:
    """Run clients in a Grid5000 deployment.

    Parameters
    ------
    nb_clients: int
        Number of clients involved in the experiments.
    storage_path: str
        Path to your network storage (ex: group storage), accessible from
        Grid5000 machines.
    data_dir: str
        Name of the directory (on the network storage) in which the data will be
        downloaded.
    roles: list[en.objects.Roles]
        Instances where the command line will be executed.
    host: str
        Address of the instance running the server.
    """
    python_bin = "/opt/dln-venv/bin/python"
    client_pool = en.get_hosts(roles=roles, pattern_hosts="client")
    client_machine = 0
    for idx in range(nb_clients):
        client_name = f"client_{idx}"
        ca_cert = os.path.join(storage_path, "ca-cert.pem")
        data_path = os.path.join(
            storage_path, data_dir, f"mnist_{DATA_SCHEME}"
        )
        serv_uri = f"wss://{host}:8765"
        background = True
        if idx == nb_clients - 1:  # last client
            background = False
        en.run_command(
            f"{python_bin} run_client.py "
            + f"--client_name {client_name} --data_folder {data_path} "
            + f"--ca_cert {ca_cert} --serv_uri {serv_uri} --verbose True",
            background=background,
            on_error_continue=True,
            roles=client_pool[client_machine],
        )
        client_machine += 1
        if client_machine >= len(client_pool):
            client_machine = 0


if __name__ == "__main__":
    fire.Fire(run_grid5k)
