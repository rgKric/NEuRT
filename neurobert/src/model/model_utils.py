import torch
from .model import BioBert, BioBertClassifier

from typing import Optional, NamedTuple
from torch import nn


def load_clean_state_dict_to_model(model_type: str, cfg: Optional[NamedTuple], checkpoint_path: str=None) -> nn.Module:
    """
    Loads a model of a given type and optionally loads a checkpoint, removing any 'module.' prefixes.

    Args:
        model_type (str): Type of the model to create. Supported: 'bert', 'classifier_for_inference', 'classifier_for_fine_tune'.
        cfg (Optional[NamedTuple]): Configuration object or dictionary for model initialization.
        checkpoint_path (str): Path to a checkpoint file to load weights from. Defaults to None.

    Returns:
        nn.Module: Initialized model with loaded weights (if checkpoint provided).

    Raises:
        TypeError: If an unsupported model_type is provided.

    Notes:
        - If checkpoint_path is provided, it removes the 'module.' prefix from keys before loading.
        - For 'bert' and 'classifier_for_inference', uses `load_state_dict`. 
          For 'classifier_for_fine_tune', uses model-specific `_load_bert`.
    """
    if model_type in ['bert'] :
        model = BioBert(cfg)
    elif model_type in ['classifier_for_inference', 'classifier_for_fine_tune']:
        model = BioBertClassifier(cfg)
    else:
        raise TypeError('No such model')
        
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        new_state_dict = {}
        for key, value in checkpoint.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
            
        if model_type in ['bert', 'classifier_for_inference']:
            model.load_state_dict(new_state_dict, )
        else:
            model._load_bert(new_state_dict)
            
    return model


def init_weights(module: nn.Module) -> None:
    """
    Initializes weights and biases of a PyTorch module.

    - Applies Xavier uniform initialization to Linear and Embedding weights (pretrain stage).
    - Sets LayerNorm weights to 1.0 and biases to 0.0.
    - Sets Linear biases to 0.0 if they exist.

    Args:
        module (nn.Module): PyTorch module to initialize.

    Returns:
        None
    """
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        torch.nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, torch.nn.Linear) and module.bias is not None:
        module.bias.data.zero_()