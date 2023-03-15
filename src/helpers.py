import pickle
import json
import numpy as np
import requests
import certifi

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
    setup main variables
    """
    # read parameters from config file
    with open(config_path, mode="rb") as file:
        config = json.load(file)
    return config