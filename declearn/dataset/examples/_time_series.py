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

#NOTE: for now the example support the creation of a dataset from one action file
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


logger = logging.getLogger(__name__)

#TODO: [] Add the documentation to the functions.
#TODO: [] Write the necessary unit tests.
#TODO: [] Make the target logic generic, for now it depends on the example dataset.

# Define the necessary types here  -------------------
Signal = list[float]

@dataclass
class EMGDatasetConfigs:
    folder : str
    actions : list[str] = field(default_factory=lambda: ACTIONS)
    target : int =  8
    subjects : list[int]= field(default_factory=lambda: list(range(0,15)) )#by default, select all the participants
    window_size : int = 128
    zip_name : str = '15Subjects-7Gestures'
    on_save: bool = True
    on_save_filename : Optional[str] = '15Subjects-1Gestures-pr-tensor'
    

#NOTE: in this example we don't care about tracking emg-subject membership
class EMGSignal:
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
    
    #TODO: basically we want to compute the length as a proprety

    @property
    def nb_windows(self, ):
        return self._nb_windows
    @nb_windows.setter
    def nb_windows(self, params):
         self._nb_windows = int(self.length / params.window_size)


    #TODO: every signal has the ability to perform sliding window routine on itself
    def _split_signal_by_window(self, params: EMGDatasetConfigs):
        if params.window_size > int(self.length /2): 
            raise ValueError("Window size must be at least the half of the total length of the signal. Please try with a smaller value")
        
        windows =  t.zeros((self.nb_windows,params.window_size))
        
        for i in range(self.nb_windows):
            windows[i, :] = t.Tensor(self.content[i*params.window_size:(i+1)*params.window_size])
        return windows
    
    def get_signal_slided_windows(self, params : EMGDatasetConfigs):
        return self._split_signal_by_window(params)
#-----------------------------------------------------
        
        
def load_semg_hand_poses(params: EMGDatasetConfigs)-> t.Tensor:
    #TODO: the function must return a dataframe containing the signal values for the target column
    #Checking for any value errors
    if not set(params.subjects).issubset(range(0,16)):
        raise ValueError(f"Invalid Subject index Value {params.subjects} Subject list must be contained in {list(range(0,15))}. Please try again.")
    
    if not (9>  params.target > 0):
        raise ValueError(f"Invalid sensor index. Target value must be between 1 and 8. Please try again")
    
    evaluate_emg_data_by_source(params)
    
    #check wether there exists an already processed .pt file
    if params.on_save_filename is not None:
        if os.path.isfile(f'{params.folder}/{params.on_save_filename}.pt'):
            return t.load(
                f'{params.folder}/{params.on_save_filename}.pt')
            
    #preprocessed file doesn't exist so we take care of that
    return get_emg_tensor(params)
    
        
def download_semg_hand_poses()-> bytes:
    logger.info("Downloading the sEMG hand poses Dataset ... ")
    
    reply = requests.get(URL, timeout=500)
    try: 
        reply.raise_for_status()
    except requests.HTTPError as exc: 
        raise RuntimeError("Failed to download sEMG hand poses dataset zip file.") from exc 
    
    return reply.content

def load_raw_dataframes(params : EMGDatasetConfigs)-> Iterator[pd.DataFrame]:
    directory = os.path.abspath(os.path.join('examples', params.folder))

    source = os.path.join(directory, f'{params.zip_name}.zip')
    zfile = zipfile.ZipFile(source)
    files = list(map(lambda x: x.filename, zfile.filelist))
    
    for s in params.subjects:
        for act in params.actions:
            current_file = f'{params.zip_name}/S{s}/emg-{act}-S{s}.csv'
            if not current_file in files: 
                continue
            with zfile.open(current_file, 'r') as z_safile:
                yield pd.read_csv(io.BytesIO(z_safile.read())) 
            
def evaluate_emg_data_by_source(params : EMGDatasetConfigs):
    
    #Assumes the opened zip file already exists
    if isinstance(params.folder, str):
        #construct folder
        directory = os.path.abspath(os.path.join('examples', params.folder))
        print(directory)
        if not os.path.isdir(directory): 
            raise ValueError(f"Failed to find {params.folder}. Please try with a valid folder.")
        
        path = os.path.join(directory, f'{params.zip_name}.zip')
        if os.path.isfile(path):
            return 
    
    #Data doesn't exists -> downloand & save data into disk (mandatory to save since we have folder for every user within the zip file)
    data = download_semg_hand_poses()
    with open(path, "wb") as file: 
        file.write(data)
        
    
def _get_normalized_emgs(params: EMGDatasetConfigs) -> Iterator[EMGSignal]:
    
    #norlmalizes the set of signals to and returns a list of EMG signals
    for df in load_raw_dataframes(params):
        
        normalized_signal : Signal = (df- df.mean()/df.std())[f'emg{params.target}'].values.tolist()
        
        emg_instance = EMGSignal(normalized_signal, params)
        
        yield emg_instance
        
        

def _split_processed_emgs_by_window(params: EMGDatasetConfigs)-> t.Tensor:
    #perform the sliding window routine and concatenate the results (Length, window_size)
    tensor = t.tensor([])
    on_save_file_path = os.path.abspath(os.path.join(f'examples', params.folder))
    on_save_file_name = f'{on_save_file_path}/{params.on_save_filename}.pt'
    
    if os.path.isfile(on_save_file_name):
        return t.load(on_save_file_name)
    
    for emg_instance in _get_normalized_emgs(params): 
        slided_windows_per_signal = emg_instance.get_signal_slided_windows(params)
        tensor = t.cat((tensor, slided_windows_per_signal), 0)
    if params.on_save:
        t.save(tensor, on_save_file_name)
    #dump the tensor into a the file
    return tensor

def get_emg_tensor(params: EMGDatasetConfigs) -> t.Tensor:
    return _split_processed_emgs_by_window(params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    
    args = parser.parse_args()
    #create the configs object
    emg_dataset_configs = EMGDatasetConfigs(folder=args.folder)
    
    #run the dataset creation script
    load_semg_hand_poses(emg_dataset_configs)
    
if __name__ == "__main__":
    main()