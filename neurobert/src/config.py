from typing import NamedTuple
from collections import namedtuple
import json

    
class MaskConfig(NamedTuple):
    """ Hyperparameters for masking """
    mask_ratio: float = 0  # masking probability
    mask_alpha: int = 0  # How many tokens to form a group.
    max_gram: int = 0  # number of max n-gram to masking
    mask_prob: float = 1.0 
    replace_prob: float = 0.0
    
    @classmethod
    def from_dict(cls, cfg):
        ''' Create config from dict '''
        return cls(**cfg)
            
    @classmethod
    def from_json(cls, file):
        ''' Load config from json file '''
        return cls(**json.load(open(file, "r")))


class ModelConfig(NamedTuple):
    """ Configuration for BERT model """
    hidden: int = 0  # Dimension of Hidden Layer in Transformer Encoder
    hidden_ff: int = 0  # Dimension of Intermediate Layers in Positionwise Feedforward Net
    feature_num: int = 0  # Factorized embedding parameterization
    n_layers: int = 0  # Numher of Hidden Layers
    n_heads: int = 0  # Numher of Heads in Multi-Headed Attention Layers
    seq_len: int = 0  # Maximum Length for Positional Embeddings

    emb_norm: bool = True
    block_embedding_mode: str = 'not_hidden' # or 'hidden'
    embedding_mode: str = 'sum_all' # or 'last'
    use_parametr_sharing_strategy: bool = False

    @classmethod
    def from_dict(cls, cfg):
        return cls(**cfg)
            
    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))
    

class TrainerConfig(NamedTuple):
    """ Hyperparameters for pretraining """
    seed: int = 0  # random seed
    batch_size: int = 0
    lr: int = 0  # encoder learning rate
    n_epochs: int = 0  # the number of epoch
    warmup: float = 0
    save_steps: int = 0  # interval for saving model

    lr_cl: float = 0.0 # classificator learning rate
    freez_mode: str = 'full' # bert freez mode
    sheduler_gamma: float = 0.95
    weight_decay: float = 0.01
    
    @classmethod
    def from_dict(cls, cfg):
        return cls(**cfg)
            
    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))