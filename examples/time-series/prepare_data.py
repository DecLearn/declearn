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

"""Data-preparation script for the Hand Poses sEMG dataset."""

import os

import fire

from declearn.dataset.examples import EMGDatasetConfigs, load_semg_hand_poses

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def prepare_data_for_clients(nb_clients: int = 2) -> None:
    """Prepares necessary data files for the clients.

    For now this example supports up to 8 clients maximum. Since the dataset is
    only composed of 8 sensors per participant, to simplify, distinction
    between client datasets is done through the alocation of a different sensor
    for every new client.

    Parameters
    ----------
    nb_clients: int
        Number of clients.

    Raises
    ------
    ValueError:
        If the number of clients surpasses the number of sensors per
        participant which is 8 for the dataset chosen for this example.
    """
    if not os.path.exists(DATA_DIR):
        # build the data folder
        os.mkdir(DATA_DIR)

    if nb_clients > 8:
        raise ValueError(
            "Number of clients exceeds the maximum "
            "number of electroids in the dataset"
        )
    for i in range(nb_clients):
        client_config = EMGDatasetConfigs(
            path=DATA_DIR,
            target=i + 1,
            on_save_filename=f"client_{i}",
        )

        load_semg_hand_poses(client_config)


if __name__ == "__main__":
    fire.Fire(prepare_data_for_clients)
