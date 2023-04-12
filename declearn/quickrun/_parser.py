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

"""
Utils parsing a data folder following a standard format into a nested
dictionnary
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from declearn.quickrun._config import DataSourceConfig

# pylint: disable=too-many-arguments,too-many-branches,too-many-locals


def parse_data_folder(
    data_config: DataSourceConfig,
    folder: Optional[str] = None,
) -> Dict:
    """Utils parsing a data folder following a standard format into a nested
    dictionnary.

    The default expected format is :

        folder/
        └─── data*/
            └─── client*/
            │      train_data.* - training data
            │      train_target.* - training labels
            │      valid_data.* - validation data
            │      valid_target.* - validation labels
            └─── client*/
            │    ...

    Parameters:
    -----------
    data_config : DataSourceConfig
        DataSourceConfig instance, see class documentation for details.
    folder : str or None
        The main experiment folder in which to look for a `data*` folder.
        Overwritten by data_folder.
    """

    data_folder = data_config.data_folder
    client_names = data_config.client_names
    dataset_names = data_config.dataset_names

    if not folder and not data_folder:
        raise ValueError(
            "Please provide either a parent folder or a data folder"
        )
    # Data_folder
    if not data_folder:
        gen_folders = Path(folder).glob("data*")  # type: ignore
        data_folder = next(gen_folders, False)  # type: ignore
        if not data_folder:
            raise ValueError(
                f"No folder starting with 'data' found in {folder}. "
                "Please store your split data under a 'data_*' folder. "
                "To use an example dataset run `declearn-split` first."
            )
        if next(gen_folders, False):
            raise ValueError(
                "More than one folder starting with 'data' found"
                f"in {folder}. Please store your data under a single"
                "parent folder"
            )
    else:
        if os.path.isdir(data_folder):
            data_folder = Path(data_folder)  # type: ignore
        else:
            raise ValueError(
                f"{data_folder} is not a valid path. To use an example "
                "dataset run `declearn-split` first."
            )
    # Get clients dir
    if client_names:
        if isinstance(client_names, list):
            valid_names = [
                os.path.isdir(os.path.join(data_folder, n))
                for n in client_names
            ]
            if sum(valid_names) != len(client_names):
                raise ValueError(
                    "Not all provided client names could be found in"
                    f"{data_folder}"
                )
            clients = {
                n: {} for n in client_names
            }  # type: Dict[str,Dict[Any,Any]]
        else:
            raise ValueError(
                "Please provide a valid list of client names for "
                "argument 'client_names'"
            )
    else:
        gen_folders = Path(data_folder).glob("client*")  # type: ignore
        first_client = next(gen_folders, False)
        if not first_client:
            raise ValueError(
                f"No folder starting with 'client' found in {data_folder}."
                " Please store your individual under client data under"
                "a 'client*' folder"
            )
        clients = {str(first_client).rsplit("/", 1)[-1]: {}}
        while client := next(gen_folders, False):
            clients[str(client).rsplit("/", 1)[-1]] = {}
    # Get train and valid files
    data_items = [
        "train_data",
        "train_target",
        "valid_data",
        "valid_target",
    ]
    if dataset_names:
        if set(data_items) != set(dataset_names.keys()):
            raise ValueError(
                f"Please provide a properly formatted dictionnary as input"
                f"using the follwoing keys : {str(data_items)}"
            )
    else:
        dataset_names = {i: i for i in data_items}
    for client, files in clients.items():  # type: ignore
        for key, val in dataset_names.items():
            filepath = Path(data_folder / client)  # type: ignore
            gen_file = filepath.glob(f"{val}*")
            file = next(gen_file, False)
            if not file:
                raise ValueError(
                    f"Could not find a file named '{val}.*' in {client}"
                )
            if next(gen_file, False):
                raise ValueError(
                    f"Found more than one file named '{val}.*' in {client}"
                )
            files[key] = str(file)
    return clients
