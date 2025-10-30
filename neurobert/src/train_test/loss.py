import torch.nn as nn
from typing import Any


def count_loss(model: nn.Module, batch: tuple, mode: str, **kwargs) -> Any:
    '''
    Computes the loss for a single batch depending on the training mode.

    Args:
        model (nn.Module): The neural network model.
        batch (tuple): Input batch. Structure depends on mode:
            - 'pretrain_reconstruction': (instance_mask, mask_pos_index, seq)
            - 'binary_classification': (instance_mask, labels)
        mode (str): Mode of operation. Must be one of ['pretrain_reconstruction', 'binary_classification', 'behavior_classification'].
        **kwargs: Additional parameters:
            - class_weights (Optional[torch.Tensor]): Weights for CrossEntropyLoss in classification modes.

    Returns:
        - For 'pretrain_reconstruction': scalar tensor representing reconstruction loss.
        - For 'binary_classification': tuple (loss tensor, model outputs).
    
    Raises:
        KeyError: If an unsupported mode is provided.
    '''
    if mode not in ['pretrain_reconstruction', 'binary_classification', 'behavior_classification']:
        raise KeyError('No such mode(')
        
    if mode == 'pretrain_reconstruction':
        criterion = nn.MSELoss(reduction='none')
        instance_mask, mask_pos_index, seq = batch
        seq_recon, _ = model(instance_mask)
        return criterion(seq_recon.gather(1, mask_pos_index.unsqueeze(2).expand(-1, -1, seq_recon.shape[-1])), seq).mean()
    
    elif mode == 'binary_classification':
        criterion = nn.CrossEntropyLoss(weight=kwargs.get('class_weights', None))
        instance_mask, labels = batch
        outputs = model(instance_mask)
        return criterion(outputs, labels), outputs