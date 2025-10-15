# coding: utf-8

# Copyright 2025 Inria (Institut National de Recherche en Informatique
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

"""Util to parse the contents of a data folder into a nested dict of paths."""

import os
from pathlib import Path
from typing import Dict, List, Optional

from declearn.quickrun._config import DataSourceConfig

__all__ = [
    "parse_data_folder",
]


def parse_data_folder(
    data_config: DataSourceConfig,
    folder: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """Parse the contents of a data folder into a nested dict of file paths.

    This function expects the folder to abide by the following standard:

        folder/
        └─── data*/
            └─── client*/
            │      train_data.* - training data
            │      train_target.* - training labels
            │      valid_data.* - validation data
            │      valid_target.* - validation labels
            └─── client*/
            │    ...

    Parameters
    ----------
    data_config: DataSourceConfig
        DataSourceConfig instance; see its documentation for details.
    folder: str or None
        The main experiment folder in which to look for a `data*` folder.
        Overridden by `data_config.data_folder` when specified.

    Returns
    -------
    paths:
        Nested directory containing the parsed file paths, with structure
        `{client_name: {file_key_name: file_path}}`, where the key names
        are always the same: "train_data", "train_target", "valid_data"
        and "valid_target".
    """
    # Identify the root data folder.
    data_folder = get_data_folder_path(data_config.data_folder, folder)
    # Identify clients' data folders.
    client_names = list_client_names(data_folder, data_config.client_names)
    clients: Dict[str, Dict[str, str]] = {c: {} for c in client_names}
    # Set up a mapping between expected files and their naming.
    data_items = [
        "train_data",
        "train_target",
        "valid_data",
        "valid_target",
    ]
    dataset_names = data_config.dataset_names
    if dataset_names:
        if set(data_items) != dataset_names.keys():
            raise ValueError(
                "Please provide a properly formatted dictionnary as input, "
                f"using the following keys: {data_items}"
            )
    else:
        dataset_names = {name: name for name in data_items}
    # Gather client-wise file paths.
    for client, paths in clients.items():
        client_dir = data_folder.joinpath(client)
        for key, val in dataset_names.items():
            files = [p for p in client_dir.glob(f"{val}*") if p.is_file()]
            if not files:
                raise ValueError(
                    f"Could not find a '{val}.*' file for client '{client}'."
                )
            if len(files) > 1:
                raise ValueError(
                    f"Found multiple '{val}.*' files for client '{client}'."
                )
            paths[key] = files[0].as_posix()
    # Return the nested directory of parsed file paths.
    return clients


def get_data_folder_path(
    data_folder: Optional[str] = None,
    root_folder: Optional[str] = None,
) -> Path:
    """Return the path to a data folder.

    Parameters
    ----------
    data_folder:
        Optional user-specified data folder.
    root_folder:
        Root folder, under which to look up a 'data*' folder.
        Unused if `data_folder` is not None.

    Returns
    -------
    dirpath:
        pathlib.Path wrapping the path to the identified data folder.

    Raises
    ------
    ValueError
        If the input arguments point to non-existing folders, or a data
        folder cannot be unambiguously found under the root folder.
    """
    # Case when a data folder is explicitly designated.
    if isinstance(data_folder, str):
        if os.path.isdir(data_folder):
            return Path(data_folder)
        raise ValueError(
            f"{data_folder} is not a valid path. To use an example "
            "dataset, run `declearn-split` first."
        )
    # Case when working from a root folder.
    if not isinstance(root_folder, str):
        raise ValueError(
            "Please provide either a data folder or its parent folder."
        )
    folders = list(Path(root_folder).glob("data*"))
    if not folders:
        raise ValueError(
            f"No folder starting with 'data' found under {root_folder}. "
            "Please store your split data under a 'data_*' folder. "
            "To use an example dataset, run `declearn-split` first."
        )
    if len(folders) > 1:
        raise ValueError(
            "More than one folder starting with 'data' found under "
            f"{root_folder}. Please store your data under a single "
            "parent folder, or specify the target data folder."
        )
    return folders[0]


def list_client_names(
    data_folder: Path,
    client_names: Optional[List[str]] = None,
) -> List[str]:
    """List client-wise subdirectories under a data folder.

    Parameters
    ----------
    data_folder:
        `pathlib.Path` designating the main data folder.
    client_names:
        Optional list of clients to restrict the outputs to.

    Raises
    ------
    ValueError
        If `client_names` is of unproper type, or lists names that cannot
        be found under `data_folder`.
    """
    # Case when client names are provided: verify that they can be found.
    if client_names:
        if not isinstance(client_names, list):
            raise ValueError(
                "Please provide a valid list of client names for "
                "argument 'client_names'"
            )
        if not all(
            data_folder.joinpath(name).is_dir() for name in client_names
        ):
            raise ValueError(
                "Not all provided client names could be found under "
                f"{data_folder}"
            )
        return client_names.copy()
    # Otherwise, list subdirectories of the data folder.
    return [path.name for path in data_folder.iterdir() if path.is_dir()]
