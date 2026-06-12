from dataclasses import dataclass
import io
import os
import zipfile
from typing import Generator, Iterator, Literal, Optional, Tuple, Union
import torch as t
import pandas as pd
import requests
import logging
import torch as t

__all__=["load_semg_hand_poses"]

#NOTE: for now the example support the creation of a dataset from one action file
ACTIONS = ['fistdown', 
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

# Define the necessary types here  -------------------
type Signal = list[float]

@dataclass
class EMGDatasetConfigs:
    folder : str
    action : str = ACTIONS[0]
    target : int =  -1
    subjects : list[int]= list(range(0,16)) #by default, select all the participants
    window_size : int = 128
    zip_name : str = '15Subjects-7-Gestures'
    on_save: bool = True
    on_save_filename : Optional[str] = '15Subjects-1-Gestures-pr-tensor'
    

#NOTE: in this example we don't care about tracking emg-subject membership
class EMGSignal:
    def __init__(self, signal : Signal): 
        self.content = signal
        self.length = self.set_length()
        self.windows = None
        self.nb_windows = 0
        
        
    #TODO: set the content in the correct type -> convert to t.Tensor
    def set_content_type(self,signal : Signal):
        #by default this function convert to pytorch tensor
        self.content = t.Tensor(signal)
    
    #TODO: basically we want to compute the length as a proprety
    def set_length(self):
        self.length = self.content.shape[0]
    
    #TODO: and hence set the number of windows it will obtain after splitting
    def set_nb_windows(self, params: EMGDatasetConfigs): 
        self.nb_windows = int(self.length / params.window_size)
        
    #TODO: every signal has the ability to perform sliding window routine on itself
    def _split_signal_by_window(self, params: EMGDatasetConfigs):
        if params.window_size > int(self.length /2): 
            raise ValueError("Window size must be at least the half of the total length of the signal. Please try with a smaller value")
        
        windows =  t.zeros((self.nb_windows,params.window_size))
        
        for i in range(params.window_size+1):
            windows[i] = t.Tensor(windows[i*params.window_size:(i+1)*params.window_size])
        return windows
    
    def get_signal_slided_windows(self, params : EMGDatasetConfigs):
        return self._split_signal_by_window(params)
#-----------------------------------------------------
        
        
def build_emg_dataset(params: EMGDatasetConfigs)-> t.Tensor:
    #TODO: the function must return a dataframe containing the signal values for the target column
    #Checking for any value errors
    if not set(params.subjects).issubset(range(0,15)):
        raise ValueError(f"Invalid Subject index Value {params.subjects} Subject list must be contained in {list(range(0,15))}. Please try again.")
    
    if (-1>  params.target > 8):
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

    source = os.path.join(params.folder, f'{params.zip_name}.zip')
    with zipfile.ZipFile(source) as zfile:

        for s in params.subjects: 
            with zfile.open(f'{source}/S{s}/emg-{params.action}-S{s}.csv', 'r') as z_safile:
                yield pd.read_csv(io.BytesIO(z_safile.read())) 
            
def evaluate_emg_data_by_source(params : EMGDatasetConfigs):
    
    #Assumes the opened zip file already exists
    if isinstance(params.folder, str): 
        if not os.path.isdir(params.folder): 
            raise ValueError(f"Failed to find {params.folder}. Please try with a valid folder.")
        
        path = os.path.join(params.folder, f'{params.zip_name}.zip')
        if os.path.isfile(path):
            return 
    
    #Data doesn't exists -> downloand & save data into disk (mandatory to save since we have folder for every user within the zip file)
    data = download_semg_hand_poses()
    with open(path, "wb") as file: 
        file.write(data)
        
    
def _get_normalized_emgs(params: EMGDatasetConfigs) -> Iterator[EMGSignal]:
    
    #norlmalizes the set of signals to and returns a list of EMG signals
    for df in load_raw_dataframes(params):
        
        normalized_signal = (df- df.mean()/df.std())[f'emg-{params.target}'].values.tolist()
        
        emg_instance = EMGSignal(normalized_signal)
        emg_instance.set_nb_windows(params)
        
        yield emg_instance
        
        

def _split_processed_emgs_by_window(params: EMGDatasetConfigs)-> t.Tensor:
    #perform the sliding window routine and concatenate the results (Length, window_size)
    tensor = t.tensor([])
    for emg_instance in _get_normalized_emgs(params): 
        slided_windows_per_signal = emg_instance.get_signal_slided_windows(params)
        tensor.cat((tensor, slided_windows_per_signal), 0)
    if params.on_save: 
        t.save(tensor, params.on_save_filename)
    #dump the tensor into a the file
    return tensor

def get_emg_tensor(params: EMGDatasetConfigs) -> t.Tensor:
    return _split_processed_emgs_by_window(params)