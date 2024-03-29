import pickle
import json
import numpy as np
import requests
import certifi
import pandas as pd
import base64
from IPython.display import display

def _read_n_fix(filename:str, filter_out=[], dropna=[], n_show=1, verbose=True) -> pd.DataFrame:
    """
    made for dirty datasets, reads dataset and returns selected preview of what's inside, plus some cleansing on the way
    filename: path to the file
    filter_out: list of columns to filter out (if nothing provided returns full dataframe)
    dropna: list of columns to drop NaN's from (if nothing provided returns full dataframe)
    n_show: number of observatiosn to show in preview of a dataframe
    verbose: verbosity (show/not shape of dataframe and first n-lines)
    return: dataframe
    """
    data = pd.read_csv(filename, dtype=str)
    cols = [col.replace(".","_").replace("/","").replace("  "," ").replace(" ","_").lower() for col in data.columns]
    data.columns = cols

    if verbose:
        print(data.shape)
        display(data.head(n_show))

    if dropna:
        data.dropna(subset=dropna, inplace=True)
        print(f"Shape after droped NaNs in {dropna}: {data.shape}")
        display(data.head(n_show))
    
    if filter_out:
        data = data.filter(filter_out)
        print(f"Shape after filtered NaNs {filter_out}: {data.shape}")
        display(data.head(n_show))

    return data

def get_base64(s):
    """
    returns base64 encoding of a string
    """
    s_ascii = s.encode("ascii")
    base64_bytes = base64.b64encode(s_ascii)
    base64_string = base64_bytes.decode("ascii")
    return base64_string

def to_pickle(file, filename):
    """"
    Saves given python object as binary file (handy to avoid problems with types etc)
    file: file object (e.g. dataframe)
    filename: saving destination (path + filename withou extention), str
    return: True (deafault)
    """
    if ".pkl" not in filename:
        filename = f"{filename}.pkl"
    with open(filename, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return True

def from_pickle(filename):
    """"
    Reads and returns a binary file 
    filename: loading destination (path + filename withou extention), str
    return: python object (whatever was saved: dict, dataframe, etc)
    """
    if ".pkl" not in filename:
        filename = f"{filename}.pkl"
    with open(filename, 'rb') as handle:
        file = pickle.load(handle)
    return file

def get_dtypes_dict(data):
    """
    returns dictionary with a columns for each of dtypes: date, object (str), numeric (number, bool)
    data: pandas dataframe
    return: dictionarys
    """
    return {'date':list(data.select_dtypes(include=np.datetime64).columns),
            'str':list(data.select_dtypes(include=np.object0).columns),
            'numeric':list(data.select_dtypes(include=[np.number, np.bool8]).columns)}

def transform_to_positive(data):
    """
    shift all observations to make them non-negative
    data: pd.Series
    return: pd.Series
    """
    return data + np.abs(data.min())

def filter_high_variability(data, min_values):
    """
    filter out only columns with minimum number of unique values (i.e. eliminate low/no variability columns)
    data: pd.DataFrame
    min_values: minimum required values, int
    return: pd.DataFrame 
    """
    unique_value_counts = data.nunique().sort_values()
    mask = unique_value_counts >= min_values
    high_variability = unique_value_counts[mask].index.to_list()
    return data.loc[:,high_variability]

def load_config(config_path:str) -> dict:
    """
    loads json into dictionary
    config_path: path to load from
    return: config dictionary
    """
    # read parameters from config file
    with open(config_path, mode="rb") as file:
        config = json.load(file)
    return config

def save_config(config, config_path:str) -> dict:
    """
    saves dictionary to json
    config_path: path to save to
    """
    # save parameters from config file
    with open(config_path, 'w') as file:
        json.dump(config, file)
    return True

def ms_to_hours(millis: int) -> str:
    """
    convert milliseconds into hours, minutes, seconds
    millis: time in milliseconds
    returns: string of hours: minutes: seconds
    """
    seconds, milliseconds = divmod(millis, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return ("%d:%d:%d" % (hours, minutes, seconds))