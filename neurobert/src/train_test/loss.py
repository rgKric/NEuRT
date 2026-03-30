import torch.nn as nn
from typing import Any
import torch


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
    if mode not in ['pretrain_reconstruction', 'binary_classification', 'multitask']:
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
    
    elif mode == 'multitask':
        weights = kwargs.get('multitask_weigths')
        criterion_cls = nn.CrossEntropyLoss(weight=kwargs.get('class_weights', None))
        criterion_mask_recon = nn.MSELoss(reduction='mean')
        creterion_full_recon = nn.MSELoss(reduction='mean')
        
        instance_mask, labels, mask_pos_index, seq = batch
        cls_out, recon_out = model(instance_mask)
        
        B, L = instance_mask.shape[:2]
        not_mask = torch.ones(B, L, dtype=torch.bool, device=instance_mask.device)
        not_mask.scatter_(1, mask_pos_index, False)
        unmusked_real = instance_mask[not_mask].view(B, L - mask_pos_index.shape[1], instance_mask.shape[2])
        unmusked_recon = recon_out[not_mask].view(B, L - mask_pos_index.shape[1], instance_mask.shape[2])
        
        loss_cls = criterion_cls(cls_out, labels) * weights[0]
        loss_mask_recon = criterion_mask_recon(recon_out.gather(1, mask_pos_index.unsqueeze(2).expand(-1, -1, recon_out.shape[-1])),
                                    seq) * weights[1] 
        loss_full_recon = creterion_full_recon(unmusked_recon, unmusked_real) * weights[2]
        
        loss = loss_cls + loss_mask_recon + loss_full_recon
        
        return loss, (cls_out, recon_out), torch.Tensor([loss_cls.item(), loss_mask_recon.item(), loss_full_recon.item()])