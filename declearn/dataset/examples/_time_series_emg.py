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
"""Utilities to load and preprocess EMGs time series datasets."""

import io
import logging
import os
import zipfile
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import pandas as pd
import requests

try:
    import torch
except ImportError as e:
    raise ImportError("Missing required module 'torch' ") from e

__all__ = ["load_semg_hand_poses", "EMGDatasetConfigs", "ACTIONS"]


logger = logging.getLogger(__name__)

ACTIONS = [
    "fistdwn",
    "fistout",
    "left",
    "neut",
    "opendwn",
    "openout",
    "right",
    "tap",
    "twodwn",
    "twout",
]
"""The list of movements done by every participant. 
Closed Hand = (fistdwn, fistout)

Open Hand = (opendwn, openout)

Victory Sign = (twodwn, twout)

Tap Action= (tap)

Wrist Extension = (right)

Wrist Flexion = (left)

Neutral = (neut)
"""

DATA_URL = "https://www.rovit.ua.es/dataset/emgs/15Subjects-7Gestures.zip"
"""Direct link to download the dataset archive, available online. """

Signal = List[float]


@dataclass
class EMGDatasetConfigs:
    """
    Configuration container for EMG dataset preprocessing and loading.

    Fields
    ------
    path : str
        Absolute path to where the data is or will be located.
    actions : list[str]
        List of action/gesture labels to keep. Default is `ACTIONS`.
    target: int
        Target EMG sensor index. Default is 8.
    subjects: list[int]
        Subject identifiers to include in the dataset. Default is all
        participants from 0 to 14.
    window_size: int
        Number of time steps per sliding window. Default is 128.
    zip_name: str
        Name of the archive or dataset bundle. Default is
        "15Subjects-7Gestures".
    on_save: bool
        Whether to save processed outputs to disk. Default is True.
    on_save_filename: Optional[str]
        Filename used when saving processed data. Default is
        "15Subjects-1Gestures-pr-tensor".
    """

    path: str
    actions: list[str] = field(default_factory=lambda: ACTIONS)
    target: int = 8
    subjects: list[int] = field(default_factory=lambda: list(range(15)))
    window_size: int = 128
    zip_name: str = "15Subjects-7Gestures"
    on_save: bool = True
    on_save_filename: Optional[str] = "15Subjects-1Gestures-pr-tensor"


class EMGSignal:
    """
    Wrapper around a single EMG signal with utilities for
    window-based splitting.

    It stores the raw signal and helper function related to time-series data.
    Subject membership is intentionally not tracked here.

    Attributes
    ----------
    content: torch.Tensor
        A tensor of size (m,n) where m is the length of the signal and N are
        the number of features.
    length: int
        The length of the tensor.
    nb_windows: int
        The number of windows that it's possible to extract from the signal
        according the provided window size.
    """

    def __init__(self, signal: Signal, window_size: int):
        """Instantiate the EMGSignal class.

        Parameters
        ----------
        signal: Signal
            Input signal introduced by the user.
        configs: EMGDatasetConfigs
            Configurations regarding the dataset
            and to be applied by on the EMG signal introduced by the user.

        """
        self.content = torch.Tensor(signal)
        self.length = self.content.shape[0]
        self.nb_windows = int(self.length / window_size)

    def get_signal_sliding_windows(self, window_size: int) -> torch.Tensor:
        """Splits the signal into N equal-length windows.

        The length of the window is a parameter introduced within configs.
        For consistency, we assume that the window size must be
        at least the half of the total length of the signal in order to
        make sure that we extract at least 2 sliding windows
        at the worst case. This method does not implement
        window overlap logic yet.

        Parameters
        ----------
        configs: EMGDatasetConfigs
            User specific processing parameters object.

        Returns
        -------
        t.Tensor:
            A matrix composed of size [N, window_size] where N is the number of
            extracted windows per the current signal.

        Raises
        ------
        ValueError:
            - If the window size is greater than half the length of the current
            signal.
        """
        if window_size > int(self.length / 2):
            raise ValueError(
                "Window size must be at least the half of the total length"
                "of the signal. Please try with a smaller value"
            )

        windows = torch.zeros((self.nb_windows, window_size))

        for i in range(self.nb_windows):
            windows[i, :] = torch.Tensor(
                self.content[i * window_size : (i + 1) * window_size]
            )
        return windows


def load_semg_hand_poses(configs: EMGDatasetConfigs) -> torch.Tensor:
    """
    Function that loads processed sEMG hand pose data as a tensor.

    The dataset used for this example is
    [EMGs datasets: Two datasets with EMGs signals of people making gestures](https://www.rovit.ua.es/dataset/emgs/)
    by (N. Nasri & al, 2019)

    The function first validates the subject list and target index. If a cached
    preprocessed tensor exists on disk, it loads and returns it. Otherwise, it
    computes the tensor from the raw EMG data and returns it eitherways.

    Parameters
    ----------
    configs: EMGDatasetConfigs
        Configuration object controlling dataset selection, preprocessing,
        and optional on-disk caching.

    Returns
    -------
    torch.Tensor:
        Processed EMG data tensor which is an [N, window_size]
        matrix where N is the total number of windows extracted.

    Raises
    ------
    ValueError:
        - If one or more subject indices are outside the valid range.
        - If `configs.target` is not a valid target index.
    """

    if not set(configs.subjects).issubset(range(0, 16)):
        raise ValueError(
            f"Invalid Subject index Value {configs.subjects}."
            f"Subject list must be contained in {list(range(0, 16))}"
        )

    if not (9 > configs.target > 0):
        raise ValueError(
            "Invalid sensor index."
            + "Target value must be between 1 and 8. Please try again"
        )

    check_emg_data_by_source(configs)

    # checks wether there exists an already processed `.pt` file
    if configs.on_save_filename is not None:
        if os.path.isfile(f"{configs.path}/{configs.on_save_filename}.pt"):
            return torch.load(f"{configs.path}/{configs.on_save_filename}.pt")

    # preprocessed file doesn't exist so we take care of that
    return get_hand_poses_emg_tensor(configs)


def check_emg_data_by_source(configs: EMGDatasetConfigs):
    """
    Checks whether the EMG dataset archive is available locally, and
    download it if not.

    The expected location is `{configs.folder}/{configs.zip_name}.zip`
    .If the archive is missing, the function downloads the dataset and stores
    it at that path.

    Parameters
    ----------
    configs: EMGDatasetConfigs
        Dataset configuration with the folder and archive name.

    Raises
    ------
    ValueError:
        If `configs.folder` does not correspond to an existing local directory.
    """
    path = os.path.join(
        os.path.abspath(configs.path), f"{configs.zip_name}.zip"
    )
    if isinstance(configs.path, str):
        directory = os.path.abspath(configs.path)
        if not os.path.isdir(directory):
            raise ValueError(
                "Pathname does not refer to an existing "
                "directory{configs.path}"
            )

        if os.path.isfile(path):
            logger.info("Loading data from existent archive file...")
            return

    data = download_semg_hand_poses()

    with open(path, "wb") as file:
        file.write(data)


def download_semg_hand_poses() -> bytes:
    """
    Function that fetches the [sEMG hand poses Dataset]((https://www.rovit.ua.es/dataset/emgs/15Subjects-7Gestures.zip))
    from the web.

    Returns
    -------
    bytes:
        Content of the file in bytes format.

    Raises
    ------
    RuntimeError
        - If encoutered an error during fetching.

    """
    logger.info("Downloading the sEMG hand poses Dataset ... ")

    reply = requests.get(DATA_URL, timeout=500)
    try:
        reply.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            "Failed to download sEMG hand poses dataset zip file."
        ) from exc

    return reply.content


def get_hand_poses_emg_tensor(
    configs: EMGDatasetConfigs,
) -> torch.Tensor:
    """
    Builds a tensor of sliding windows from all normalized EMG signals.

    This function loads normalized EMG signals for the selected
    subjects, splits each signal into fixed-size windows,
    concatenates all windows into a single tensor,
    and optionally saves the result to disk. If a cached
    tensor already exists, it is loaded and returned by default.

    If `configs.on_save` is True, the resulting tensor is saved to
    `{configs.path}/{configs.on_save_filename}.pt}`.

    Parameters
    ----------
    configs: EMGDatasetConfigs
        Dataset configuration containing the folder, window size,
        output filename, and caching options.

    Returns
    -------
    torch.Tensor:
        Tensor object containing the concatenated sliding windows from
        all subjects.

    """
    tensor = torch.tensor([])
    on_save_file_path = os.path.abspath(configs.path)
    on_save_file_name = f"{on_save_file_path}/{configs.on_save_filename}.pt"

    if os.path.isfile(on_save_file_name):
        return torch.load(on_save_file_name)

    for emg_instance in get_normalized_emgs(configs):
        slided_windows_per_signal = emg_instance.get_signal_sliding_windows(
            configs.window_size
        )
        tensor = torch.cat((tensor, slided_windows_per_signal), 0)
    if configs.on_save:
        torch.save(tensor, on_save_file_name)
    # dump the tensor into a the file
    return tensor


def get_normalized_emgs(configs: EMGDatasetConfigs) -> Iterator[EMGSignal]:
    """Generator function that loads, normalizes then yields the signal in a
    lazy manner.

    The function lazy loads raw dataframes of the raw signals. Applies a
    [Standard Score Normalization](https://en.wikipedia.org/wiki/Standard_score)
    on every obtained signal to create an valid EMGSignal instance.

    Parameters
    ----------
    configs: EMGDatasetConfigs
        Dataset configuration specifying the archive location, selected
        subjects, and selected actions.

    Yields
    ------
        Iterator: EMGSignal
            EMGSignal instance contained the normalized time-series.
    """

    for df in load_raw_dataframes(configs):
        normalized_signal: Signal = (df - df.mean() / df.std())[
            f"emg{configs.target}"
        ].values.tolist()

        emg_instance = EMGSignal(normalized_signal, configs.window_size)

        yield emg_instance


def load_raw_dataframes(configs: EMGDatasetConfigs) -> Iterator[pd.DataFrame]:
    """Yields raw EMG data frames from a zipped dataset.

    The archive is expected to contain files organized as:
    `zip_name/S{subject}/emg-{action}-S{subject}.csv` as specified
    in the [sEMG hand poses dataset](https://www.rovit.ua.es/dataset/emgs)
    data folder.

    To avoid unnecessary errors, files that are not present in the archive are
    skipped. Please refer to the dataset documentation for further information.

    Parameters
    ----------
    configs: EMGDatasetConfigs
        Dataset configuration specifying the archive location, selected
        subjects, and selected actions.

    Yields
    ------
    Iterator: pd.DataFrame
        One raw EMG recording loaded from a CSV file.
    """

    directory = os.path.abspath(configs.path)

    source = os.path.join(directory, f"{configs.zip_name}.zip")
    zfile = zipfile.ZipFile(source)
    files = list(map(lambda x: x.filename, zfile.filelist))

    for s in configs.subjects:
        for act in configs.actions:
            current_file = f"{configs.zip_name}/S{s}/emg-{act}-S{s}.csv"

            # skip files that are not found
            if current_file not in files:
                continue
            with zfile.open(current_file, "r") as z_safile:
                yield pd.read_csv(io.BytesIO(z_safile.read()))
