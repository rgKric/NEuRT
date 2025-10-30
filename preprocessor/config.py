from typing import NamedTuple
import json

class DataLoaderConfig(NamedTuple):
    """ Parameters for data loading """
    path: str = ''  # path to datset
    key_activity: list = ['', '']   # a list or tuple containing the keys to access the activity CSV file path in the metadata
    key_centroids: list = ['', '']  # a list or tuple containing the keys to access the centroids CSV file path in the metadata
    sep: str = ','   # the delimiter used in the CSV files
    header: None = None
    index_col: None = None
    
    @classmethod
    def from_dict(cls, cfg):
        ''' Create config from dict '''
        return cls(**cfg)
            
    @classmethod
    def from_json(cls, file):
        ''' Load config from json file '''
        return cls(**json.load(open(file, "r")))
    

class PreprocessorConfig(NamedTuple):
    """ Parametrs for data preprocessing """        
    is_norm: bool = False   # is need norm data
    delta: float = 0.0   # delta (model hiperparametr)
    length: int = 0   # length of sequence
    number: int = 0   # number of sequnses created for one neuron
    
    @classmethod
    def from_dict(cls, cfg):
        ''' Create config from dict '''
        return cls(**cfg)
            
    @classmethod
    def from_json(cls, file):
        ''' Load config from json file '''
        return cls(**json.load(open(file, "r")))
    

class FilterConfig(NamedTuple):
    "Keys for <centroid>.csv"
    id_col: str = ''
    x_col: str = ''
    y_col: str = ''

    @classmethod
    def from_dict(cls, cfg):
        ''' Create config from dict '''
        return cls(**cfg)
            
    @classmethod
    def from_json(cls, file):
        ''' Load config from json file '''
        return cls(**json.load(open(file, "r")))