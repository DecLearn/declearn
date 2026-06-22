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
from dataclasses import dataclass, field
import io
import os
import zipfile
from typing import Iterator, Optional
import torch as t
import pandas as pd
import requests
import logging
import torch as t 
import argparse

__all__=[
    "load_semg_hand_poses",
    "EMGDatasetConfigs"
    ]

ACTIONS = ['fistdwn', 
           'fistout', 
           'left', 
           'neut', 
           'opendwn', 
           'openout',
           'right', 
           'tap', 
           'twodwn',
           "twout"
           ]
URL = "https://www.rovit.ua.es/dataset/emgs/15Subjects-7Gestures.zip"

# enable logging logger.info messages 
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

Signal = list[float]

@dataclass
class EMGDatasetConfigs:
    """
        Configuration container for EMG dataset preprocessing and loading.

        Attributes:
            folder (str): Root folder containing the EMG data.
            actions (list[str]): List of action/gesture labels to keep.
            target (int): Target emg sensor index.
            subjects (list[int]): Subject identifiers to include in the dataset.
            window_size (int): Number of time steps per sliding window.
            zip_name (str): Name of the archive or dataset bundle.
            on_save (bool): Whether to save processed outputs to disk.
            on_save_filename (Optional[str]): Filename used when saving processed data.
    """
    folder : str
    actions : list[str] = field(default_factory=lambda: ACTIONS)
    target : int =  8
    subjects : list[int]= field(default_factory=lambda: list(range(0,15)) ) #by default, select all the participants
    window_size : int = 128
    zip_name : str = '15Subjects-7Gestures'
    on_save: bool = True
    on_save_filename : Optional[str] = '15Subjects-1Gestures-pr-tensor'
    

#NOTE: in this example we don't care about tracking emg-subject membership
class EMGSignal:
    """
    Wrapper around a single EMG signal with utilities for window-based splitting.

    It stores the raw signal and helper function related to time-series data.
    Subject membership is intentionally not tracked here.

    Args:
        signal (Signal): Raw EMG signal data.
        params (EMGDatasetConfigs): Dataset configuration used to define
            window size and other preprocessing settings.
    """
    def __init__(self, signal : Signal, params : EMGDatasetConfigs):
        self.length = 0
        self.content = signal
        self.nb_windows = params
        
    @property
    def content(self, ):
        return self._content
    @content.setter
    def content(self,signal):
        self._content = t.Tensor(signal)
        self.length = self._content.shape[0]
    
    #basically we want to compute the length as a proprety
    @property
    def nb_windows(self, ):
        return self._nb_windows
    @nb_windows.setter
    def nb_windows(self, params):
         self._nb_windows = int(self.length / params.window_size)


    #every signal has the ability to perform sliding window routine on itself
    def _split_signal_by_window(self, params: EMGDatasetConfigs) -> t.Tensor:
        """ Private method which splits the signal into N sliding windows. The legth of the window is a parameter introduced within params.
            For consistency, we assume that the window size must be at least the half of the total length of the signal in order to
            make sure that we extract at least 2 sliding windows at the worst case. This method does not implement
            window overlap logic yet.
        

        Args:
            params (EMGDatasetConfigs): processing parameter object.

        Raises:
            ValueError: if the window size is greater than half the length of the current signal.

        Returns:
            t.Tensor: a matrix composed of size [N, window_size] where N is the number of extracted windows per the current signal.
        """
        if params.window_size > int(self.length /2): 
            raise ValueError("Window size must be at least the half of the total length of the signal. Please try with a smaller value")
        
        windows =  t.zeros((self.nb_windows,params.window_size))
        
        for i in range(self.nb_windows):
            windows[i, :] = t.Tensor(self.content[i*params.window_size:(i+1)*params.window_size])
        return windows
    
    def get_signal_sliding_windows(self, params : EMGDatasetConfigs)-> t.Tensor:
        """ Wrapper method that utilises private method _split_signal_by_window the EMG signal split into fixed-size sliding windows.
            This wrapper method server as the public access to split the signal. 

        Args:
            params (EMGDatasetConfigs): object containing desired parameters to apply on the signal processing.

        Returns:
            t.Tensor: a matrix composed of size [N, window_size] where N is the number of extracted windows per the current signal.
        """
        return self._split_signal_by_window(params)
        
        
def load_semg_hand_poses(params: EMGDatasetConfigs)-> t.Tensor:
    """
    Function that loads processed sEMG hand pose data as a tensor. The dataset used for this example is 
    [EMGs datasets: Two datasets with EMGs signals of people making gestures](https://www.rovit.ua.es/dataset/emgs/) by (N. Nasri & al, 2019)

    The function first validates the subject list and target index. If a cached
    preprocessed tensor exists on disk, it loads and returns it. Otherwise, it
    computes the tensor from the raw EMG data and returns it eitherways. 

    Args:
        params (EMGDatasetConfigs): Configuration object controlling dataset
            selection, preprocessing, and optional on-disk caching.

    Raises:
        ValueError: If one or more subject indices are outside the valid range.
        ValueError: If `params.target` is not a valid target index.

    Returns:
        t.Tensor: Processed EMG data tensor which is an [N, window_size] matrix where N is the total number of windows extracted.
    """

    #Checking for any value errors
    if not set(params.subjects).issubset(range(0,16)):
        raise ValueError(f"Invalid Subject index Value {params.subjects} Subject list must be contained in {list(range(0,15))}. Please try again.")
    
    if not (9>  params.target > 0):
        raise ValueError(f"Invalid sensor index. Target value must be between 1 and 8. Please try again")
    
    evaluate_emg_data_by_source(params)
    
    #check wether there exists an already processed `.pt` file
    if params.on_save_filename is not None:
        if os.path.isfile(f'{params.folder}/{params.on_save_filename}.pt'):
            return t.load(
                f'{params.folder}/{params.on_save_filename}.pt')
            
    #preprocessed file doesn't exist so we take care of that
    return get_hand_poses_emg_tensor(params)
    
        
def download_semg_hand_poses()-> bytes:
    """ Function that fetches the [sEMG hand poses Dataset]((https://www.rovit.ua.es/dataset/emgs/15Subjects-7Gestures.zip))
    from the web.

    Raises:
        RuntimeError: if encouters an error during fetching.

    Returns:
        bytes: content of the file in bytes format.
    """
    logger.info("Downloading the sEMG hand poses Dataset ... ")
    
    reply = requests.get(URL, timeout=500)
    try: 
        reply.raise_for_status()
    except requests.HTTPError as exc: 
        raise RuntimeError("Failed to download sEMG hand poses dataset zip file.") from exc 
    
    return reply.content

def load_raw_dataframes(params : EMGDatasetConfigs)-> Iterator[pd.DataFrame]:
    """Yield raw EMG data frames from a zipped dataset.

    The archive is expected to contain files organized as:
    `zip_name/S{subject}/emg-{action}-S{subject}.csv` as found in the [sEMG hand poses dataset](https://www.rovit.ua.es/dataset/emgs) datafolder.

    Args:
        params (EMGDatasetConfigs): Dataset configuration specifying the archive
            location, selected subjects, and selected actions.

    Yields:
        pd.DataFrame: One raw EMG recording loaded from a CSV file.

    Notes:
        To avoid unnecessary errors, files that are not present in the archive are skipped.
    """
    
    directory = os.path.abspath(os.path.join('examples', params.folder))

    source = os.path.join(directory, f'{params.zip_name}.zip')
    zfile = zipfile.ZipFile(source)
    files = list(map(lambda x: x.filename, zfile.filelist))
    
    for s in params.subjects:
        for act in params.actions:
            current_file = f'{params.zip_name}/S{s}/emg-{act}-S{s}.csv'
            
            #skip files that are not found
            if not current_file in files: 
                continue
            with zfile.open(current_file, 'r') as z_safile:
                yield pd.read_csv(io.BytesIO(z_safile.read())) 
            
def evaluate_emg_data_by_source(params : EMGDatasetConfigs):
    """
        Check whether the EMG dataset archive is available locally, and download it if not.

        The expected location is `examples/{params.folder}/{params.zip_name}.zip`.
        If the archive is missing, the function downloads the dataset and stores it
        at that path.

        Args:
            params (EMGDatasetConfigs): Dataset configuration with the folder name
                and archive name.

        Raises:
            ValueError: If `params.folder` does not correspond to an existing local directory.
    """
    #Assumes the opened zip file already exists
    if isinstance(params.folder, str):
        #construct folder
        directory = os.path.abspath(os.path.join('examples', params.folder))
        if not os.path.isdir(directory): 
            raise ValueError(f"Failed to find {params.folder}. Please try with a valid folder.")
        
        path = os.path.join(directory, f'{params.zip_name}.zip')
        if os.path.isfile(path):
            return 
    
    #Data doesn't exist -> downloand & save into disk (mandatory to save since we have tree structure of folders for every participant within the zip file)
    data = download_semg_hand_poses()
    with open(path, "wb") as file: 
        file.write(data)
        
    
def _get_normalized_emgs(params: EMGDatasetConfigs) -> Iterator[EMGSignal]:
    
    #norlmalizes the set of signals to and returns a list of EMG signals
    for df in load_raw_dataframes(params):
        
        normalized_signal : Signal = (df- df.mean()/df.std())[f'emg{params.target}'].values.tolist()
        
        emg_instance = EMGSignal(normalized_signal, params)
        
        yield emg_instance
        
        

def _concatenate_extracted_windows_from_all_subjects(params: EMGDatasetConfigs)-> t.Tensor:
    """
    Build a tensor of sliding windows from all normalized EMG signals.

    This private function loads normalized EMG signals for the selected subjects,
    splits each signal into fixed-size windows, concatenates all windows into
    a single tensor, and optionally saves the result to disk. If a cached
    tensor already exists, it is loaded and returned immediately.

    Args:
        params (EMGDatasetConfigs): Dataset configuration containing the folder,
            window size, output filename, and caching options.

    Returns:
        t.Tensor: Tensor containing the concatenated sliding windows from all
        subjects.

    Notes:
        If `params.on_save` is True, the resulting tensor is saved to
        `{examples/{params.folder}/{params.on_save_filename}.pt}`.
    """
    #perform the sliding window routine and concatenate the results (Length, window_size)
    tensor = t.tensor([])
    on_save_file_path = os.path.abspath(os.path.join(f'examples', params.folder))
    on_save_file_name = f'{on_save_file_path}/{params.on_save_filename}.pt'
    
    if os.path.isfile(on_save_file_name):
        return t.load(on_save_file_name)
    
    for emg_instance in _get_normalized_emgs(params): 
        slided_windows_per_signal = emg_instance.get_signal_sliding_windows(params)
        tensor = t.cat((tensor, slided_windows_per_signal), 0)
    if params.on_save:
        t.save(tensor, on_save_file_name)
    #dump the tensor into a the file
    return tensor

def get_hand_poses_emg_tensor(params: EMGDatasetConfigs) -> t.Tensor:
    """
    Load or compute the hand-poses EMG tensor for all selected subjects.

    This is the public entry point for obtaining the preprocessed EMG tensor.
    It computes sliding windows from all normalized EMG signals, concatenates
    them, and optionally caches the result to disk. If a cached tensor already
    exists, it is loaded and returned.

    Args:
        params (EMGDatasetConfigs): Dataset configuration containing the folder,
            window size, output filename, and caching options.

    Returns:
        t.Tensor: Tensor containing the concatenated sliding windows from all
            selected subjects.
    Note: 
        For the sake of simplicity, the example support the creation of a dataset from one action file

    """
    return _concatenate_extracted_windows_from_all_subjects(params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    
    logger.info("Loading sEMG hand poses dataset ...")
    args = parser.parse_args()
    
    #create the configs object
    emg_dataset_configs = EMGDatasetConfigs(folder=args.folder)
    
    #run the dataset creation script
    load_semg_hand_poses(emg_dataset_configs)
    logger.info('Successfully loaded dataset!')
if __name__ == "__main__":
    main()