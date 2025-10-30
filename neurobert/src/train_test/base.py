import os

import numpy as np

from torch import from_numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.nn.functional import cosine_similarity
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix
from .loss import count_loss

from typing import Optional, Any
from collections.abc import Callable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from tqdm.auto import tqdm


class Trainer:
    def __init__(self,
                 model: nn.Module, 
                 optimizer: Optimizer,
                 device: torch.device,
                 n_epochs: int,
                 save_steps: int=1,
                 scheduler: _LRScheduler=None,
                 save_path: str=None):
        '''
        Initializes the training class.

        Args:
            model (nn.Module): The neural network model to train.
            optimizer (Optimizer): Optimizer for updating model parameters.
            device (torch.device): Device to use for training ('cpu' or 'cuda').
            n_epochs (int): Number of training epochs.
            save_steps (int): Interval of epochs to save the model. Defaults to 1.
            scheduler (_LRScheduler): Scheduler. Defaults to None.
            save_path (str): Directory path to save model checkpoints. Defaults to None.
        '''
        self.model = model
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.save_steps = save_steps
        self.save_path = save_path
        self.device = device # device name
        self.scheduler = scheduler


    def _prepare_model(self, data_parallel: bool) -> None:
        '''
        Prepares the model for training by moving it to the appropriate device.
        If `data_parallel` is True and the model is not already wrapped in `nn.DataParallel`,
        it checks the number of available GPUs and wraps the model for multi-GPU training if possible.
        
        Args:
            data_parallel (bool): Whether to enable multi-GPU training using `nn.DataParallel`.
            
        Returns:
            None
        '''
        if data_parallel and  not isinstance(self.model, nn.DataParallel):
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs for training")
                self.model = nn.DataParallel(self.model)
            else:
                print("DataParallel is set to True, but only one GPU is available. Training will use a single GPU.")
        self.model = self.model.to(self.device)


    def _save_model(self, epoch: int) -> None:
        '''
        Saves the model's state to disk at specified intervals.
        The model is saved only if `save_steps` is reached and `save_path` is defined.
        
        Args:
            epoch (int): The current training epoch, used to determine save frequency and filename.
            
        Returns:
            None
        '''
        if (epoch + 1) % self.save_steps == 0 and self.save_path is not None:
            print("Save_model")
            torch.save(self.model.state_dict(), os.path.join(self.save_path, f'model_epoch_{epoch + 1}.pth'))
            
    def _save_loggs(self, mode, train_losses, val_losses, train_loggs, val_loggs) -> None:
        '''
        ПРОПИСАТЬ СОХРАНЕНИЕ ЛОССОВ И МЕТРИК
        '''
        if train_losses:
            np.savez(os.path.join(self.save_path, 'TRAIN_LOSS.npz'), train_losses=train_losses)
        else:
            print('No Train losses')
            
        if val_losses:
            np.savez(os.path.join(self.save_path, 'VAL_LOSS.npz'), val_losses=val_losses)
        else:
            print('No Val losses')

        if mode == 'binary_classification':
            if train_loggs:
                np.savez(os.path.join(self.save_path, 'TRAIN_METRICS.npz'), np.sum(train_loggs, axis=1))
            else:
                print('No Train metrics')

            if val_loggs:
                if isinstance(val_loggs[0], tuple):
                    cm_val_epochs = [val_loggs[i][0] for i in range(len(val_loggs))]
                    keys = val_loggs[0][1].keys()
                    values = np.array([list(val_loggs[i][1].values()) for i in range(len(val_loggs))]).T
                    sessions_probs = dict(zip(keys, values))
                    np.savez(os.path.join(self.save_path, 'VAL_METRICS.npz'), cm_val_epochs)
                    np.savez(os.path.join(self.save_path, 'SESSION_PROBS.npz'), sessions_probs)
                else:
                    np.savez(os.path.join(self.save_path, 'VAL_METRICS.npz'), val_loggs)
            else:
                print('No Val metrics')
    

    def _batch_iteration(self, 
                         mode: str, 
                         func_loss: Callable=count_loss, 
                         *args, 
                         **kwargs) -> tuple:
        '''
        Performs a single batch iteration and updates model weights depending on the training mode.
        
        This method handles different modes of training:
            - 'pretrain_reconstruction': computes reconstruction loss for pretraining.
            - 'binary_classification' or 'behavior_classification': computes classification loss,
            performs a backward pass, and optionally returns a confusion matrix.
        
        Args:
            mode (str): The current training mode, determines processing logic.
            func_loss (callable): Loss function to compute the batch loss. Defaults to `count_loss`.
            *args (tuple): Input batch elements, number depends on mode:
                - 'pretrain_reconstruction': (instance_mask, mask_pos_index, seq)
                - classification modes: (instance_mask, labels, session_idx)
            **kwargs: Additional parameters:
                - class_labels (list[str]): Required for computing confusion matrices in classification modes.
                - class_weights (ndarray of shape (n_classes,)): Optional weights for cross-entropy loss.
        
        Returns:
            tuple: 
                - For pretraining: (None, loss_value)
                - For classification: (confusion_matrix, loss_value)
        
        Raises:
            TypeError: If `func_loss` is not callable.
            ValueError: If the number of batch elements does not match expected mode requirements,
                        or if required kwargs are missing.
            KeyError: If an unsupported mode is provided.
        '''
        if not callable(func_loss):
            raise TypeError("func_loss must be callable")
            
        if mode == 'pretrain_reconstruction':
            if len(args) != 3:
                raise ValueError("Expected 3 elements in batch for 'pretrain_reconstruction'")
                
            instance_mask, mask_pos_index, seq = args
            instance_mask = instance_mask.to(self.device)
            mask_pos_index = mask_pos_index.to(self.device)
            seq = seq.to(self.device)
            batch = (instance_mask, mask_pos_index, seq)
            self.optimizer.zero_grad()

            loss = func_loss(self.model, batch, mode, **kwargs)
            loss.backward()
            self.optimizer.step()
            
            return None, loss.item()
        
        elif mode in ['binary_classification', 'behavior_classification']:
            if len(args) != 3:
                raise ValueError(f"Expected 3 elements in batch for mode '{mode}'")
            if 'class_labels' not in kwargs or kwargs['class_labels'] is None:
                raise ValueError("class_labels is required to compute confusion matrix")
            
            instance_mask, labels, session_idx = args
            instance_mask = instance_mask.to(self.device)
            labels = labels.to(self.device)
            batch = (instance_mask, labels)
            self.optimizer.zero_grad()

            loss, outputs = func_loss(self.model, batch, mode, **kwargs)
            loss.backward()
            self.optimizer.step()
        
            if mode == 'binary_classification':
                preds = torch.argmax(outputs.detach().cpu(), dim=1)
            else:
                preds = torch.argmax(outputs.detach().cpu(), dim=2)
            cm = confusion_matrix(labels.detach().cpu().flatten(), preds.flatten(), labels=kwargs.get('class_labels'))
            return cm, loss.item()
        
        else:
            raise KeyError('No such mode(')
        

    def train(self, 
          mode: str,  
          data_loader_train: DataLoader, 
          func_loss: Callable=count_loss,
          data_loader_val: Optional[DataLoader]=None, 
          data_parallel: bool=False, 
          class_labels: Optional[list[int]]=None, 
          class_weights_train: Optional[torch.Tensor]=None,
          class_weights_val: Optional[torch.Tensor]=None,
          session_probs: bool=False) -> tuple[list, list, list, list]:
        '''
        General training loop for various tasks including pretraining and classification.

        This method handles the entire training process:
            - Prepares the model for training (single or multi-GPU).
            - Iterates through training batches, computes loss, and updates model weights.
            - Optionally evaluates on a validation dataset.
            - Applies learning rate scheduler if provided.
            - Saves model checkpoints and logs.

        Args:
            mode (str): Training mode. Must be one of ['pretrain_reconstruction', 'binary_classification', 'behavior_classification'].
            data_loader_train (DataLoader): PyTorch DataLoader providing training batches.
            func_loss (callable): Loss function to compute batch loss. Defaults to `count_loss`.
            data_loader_val (Optional[DataLoader]): Optional DataLoader for validation.
            data_parallel (bool): Whether to use multiple GPUs via `nn.DataParallel`.
            class_labels (Optional[list[int]]): Required for classification modes to compute confusion matrices.
            class_weights_train (Optional[torch.Tensor]): Optional class weights for training.
            class_weights_val (Optional[torch.Tensor]): Optional class weights for validation.
            session_probs (bool): If True, returns session-level probabilities during validation (classification tasks).

        Returns:
            tuple[list, list, list, list]:
                - train_losses: Average training loss per epoch.
                - val_losses: Average validation loss per epoch (if validation is used).
                - train_epochs_loggs: Batch-level logs per epoch for training.
                - val_epoch_loggs: Batch-level logs per epoch for validation (if validation is used).

        Raises:
            KeyError: If an unsupported mode is provided or class_labels are missing for classification.
            TypeError: If class weights are of unsupported type.
        '''
        if mode not in ['pretrain_reconstruction', 'binary_classification', 'behavior_classification']:
            raise KeyError('No such mode(')
        if mode in ['binary_classification', 'behavior_classification'] and class_labels == None:
            raise KeyError('Need labels of classes for this mode')
        self._prepare_model(data_parallel)
        
        if isinstance(class_weights_train, (list, tuple, np.ndarray, torch.Tensor)):
            class_weights_train = torch.tensor(class_weights_train, dtype=torch.float32).to(self.device)
        elif class_weights_train != None:
            raise TypeError("class_weights_train must be a list, tuple, np.ndarray, torch.Tensor, or None")
            
        if isinstance(class_weights_val, (list, tuple, np.ndarray, torch.Tensor)):
            class_weights_val = torch.tensor(class_weights_val, dtype=torch.float32).to(self.device)
        elif class_weights_val != None:
            raise TypeError("class_weights_val must be a list, tuple, np.ndarray, torch.Tensor, or None")
        
        train_losses = []
        val_losses = []
        train_epochs_loggs = []
        val_epoch_loggs = []
        
        # train loop
        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0
            train_batch_loggs = []
            for batch in tqdm(data_loader_train, desc=f'Epoch: {epoch + 1}'):
                train_loggs, loss_value = self._batch_iteration(mode, 
                                                           func_loss, 
                                                           *batch, 
                                                           class_labels=class_labels, 
                                                           class_weights=class_weights_train)
                epoch_loss += loss_value
                train_batch_loggs.append(train_loggs)
                
            if self.scheduler:
                self.scheduler.step() # (ExponentialLR)
            
            if mode != 'pretrain_reconstruction':
                train_epochs_loggs.append(train_batch_loggs)
            avg_train_loss = epoch_loss / len(data_loader_train)
            train_losses.append(avg_train_loss)
            print(f"Epoch {epoch + 1}/{self.n_epochs}, Training Loss: {avg_train_loss:.4f}")

            # validation
            if data_loader_val is not None:
                tester = Tester(self.model, data_loader_val, self.device)
                val_loggs, val_loss = tester.test(func_loss=func_loss, 
                                                  class_labels=class_labels, 
                                                  session_probs=session_probs,
                                                  class_weights=class_weights_val,
                                                  mode=mode)
                val_losses.append(val_loss)
                val_epoch_loggs.append(val_loggs)
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Validation Loss: {val_loss:.4f}")

            # Save the model checkpoint
            self._save_model(epoch)
            # Save full info
            self._save_loggs(mode, train_losses, val_losses, train_epochs_loggs, val_epoch_loggs)
            
        return train_losses, val_losses, train_epochs_loggs, val_epoch_loggs

    
class Tester:
    def __init__(self, 
                 model: nn.Module, 
                 loader: DataLoader, 
                 device: torch.device, 
                 save_path: str=None):
        '''
        Initialize a Tester instance for evaluating models.

        Args:
            model (nn.Module): PyTorch model to evaluate.
            loader (DataLoader): DataLoader for validation dataset.
            device (torch.device): Device to run evaluation on (CPU or GPU).
            save_path (Optional[str]): Path to save metrics or logs.
        '''
        self.model = model
        self.loader = loader
        self.device = device
        self.save_path = save_path
        

    def _prepare_model(self, data_parallel: bool) -> None:
        '''
        Prepares the model for training by moving it to the appropriate device.
        If `data_parallel` is True and the model is not already wrapped in `nn.DataParallel`,
        it checks the number of available GPUs and wraps the model for multi-GPU training if possible.
        
        Args:
            data_parallel (bool): Whether to enable multi-GPU training using `nn.DataParallel`.
            
        Returns:
            None
        '''
        if data_parallel and  not isinstance(self.model, nn.DataParallel):
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs for training")
                self.model = nn.DataParallel(self.model)
            else:
                print("DataParallel is set to True, but only one GPU is available. Training will use a single GPU.")
        self.model = self.model.to(self.device)
        

    def _save_loggs(self, mode: str=None, metrics: Any=None) -> None:
        '''
        Save evaluation metrics depending on mode.

        Args:
            mode (Optional[str]): 'pretrain_reconstruction' or classification mode.
            metrics (Optional[Any]): 
                - dict for pretraining (keys like 'r2', 'MAE', 'cosine_similarity_mean/std')
                - tuple(cm, session_probs) or array for classification

        Returns:
            None
        '''
        if mode == 'pretrain_reconstruction':
            np.savez(os.path.join(self.save_path, 'METRICS.npz'), **metrics)

        elif mode == 'binary_classification':
            if isinstance(metrics, tuple):
                cm, session_probs = metrics
                np.save(os.path.join(self.save_path, 'CONFUSION_MATRIX.npy'), cm)
                np.savez(os.path.join(self.save_path, 'SESSION_PROBS.npz'), **session_probs)
            else:
                np.save(os.path.join(self.save_path, 'CONFUSION_MATRIX.npy'), metrics)


    def _evaluate_batch(self, 
                        mode: str, 
                        func_loss: Callable, 
                        *args: Any, 
                        **kwargs: Any) -> tuple[Any, float]:
        '''
        Evaluate a single batch and collect model outputs and loss.

        Args:
            mode (str): Mode of evaluation ('pretrain_reconstruction', 'binary_classification', 'behavior_classification').
            func_loss (Optional[Callable]): Loss function, can be None for inference only.
            *args: Positional arguments representing a batch.
                - pretrain_reconstruction: (instance_mask: torch.Tensor[seq_len, features],
                                             mask_pos_index: torch.Tensor[mask_len],
                                             seq: torch.Tensor[mask_len, features])
                - classification: (instance_mask: torch.Tensor[batch, features], 
                                   labels: torch.Tensor[batch] or [batch, seq_len],
                                   session_idx: torch.Tensor[batch])
            **kwargs: Keyword args, e.g., class_labels or session_probs.

        Returns:
            tuple:
                - outputs: 
                    - pretrain_reconstruction: tuple[torch.Tensor[batch, features], torch.Tensor[batch, features], torch.Tensor[batch, features]]
                    - classification: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                - loss_value: float
        '''
        if not callable(func_loss) and func_loss != None:
            raise TypeError("func_loss must be callable or None")
        
        if mode == 'pretrain_reconstruction':
            instance_mask, mask_pos_index, seq = args
            instance_mask = instance_mask.to(self.device)
            mask_pos_index = mask_pos_index.to(self.device)
            seq = seq.to(self.device)

            if func_loss is not None:
                batch = (instance_mask, mask_pos_index, seq)
                loss = func_loss(self.model, batch, mode, **kwargs)
                loss = loss.mean()
                loss_value = loss.item()
            else:
                loss_value = 0.0
                
            seq_recon, _ = self.model(instance_mask)
            preds = seq_recon.gather(1, mask_pos_index.unsqueeze(2).expand(-1, -1, seq_recon.shape[-1]))  
            cos_sim = cosine_similarity(preds.detach().cpu(), seq.detach().cpu(), dim=1)
            
            return (preds.detach().cpu(), seq.detach().cpu(), cos_sim), loss_value
            
        elif mode in ['binary_classification', 'behavior_classification']:
            instance_mask, labels, session_idx = args
            instance_mask = instance_mask.to(self.device)
            labels = labels.to(self.device)

            if func_loss is not None:
                batch = (instance_mask, labels)
                loss, _ = func_loss(self.model, batch, mode, **kwargs)
                loss_value = loss.item()
            else:
                loss_value = 0.0
                
            outputs = self.model(instance_mask)
            if mode == 'binary_classification':
                preds = torch.argmax(outputs, dim=1).detach().cpu()
                labels = labels.cpu()
            else:
                preds = torch.argmax(outputs, dim=2).detach().cpu()
            return (preds, labels, session_idx), loss_value
        
        else:
            raise  KeyError('No such mode(')
            

    def _count_statistics(self, 
                          mode: str, 
                          eval_data: list[Any], 
                          **kwargs: Any):
        '''
        Compute statistics on validation data.

        Args:
            mode (str): Evaluation mode.
            eval_data (List[Any]): List of batch outputs from _evaluate_batch.
            **kwargs: class_labels (for confusion_matrix) or session_probs flag.

        Returns:
            - pretrain_reconstruction: dict with keys: 'r2', 'MAE', 'cosine_similarity_mean', 'cosine_similarity_std'
            - classification: np.ndarray confusion_matrix or tuple[confusion_matrix, dict session_probs]
        '''
        if mode == 'pretrain_reconstruction':
            preds = [eval_data[i][0] for i in range(len(eval_data))]
            seqs = [eval_data[i][1] for i in range(len(eval_data))]
            cos_sim = [eval_data[i][2] for i in range(len(eval_data))]
            
            seqs = torch.cat(seqs, dim=0).view(-1, 5).cpu().numpy()
            preds = torch.cat(preds, dim=0).view(-1, 5).cpu().numpy()
            cos_sim = torch.cat(cos_sim, dim=0).cpu().numpy()
            
            r2 = r2_score(seqs, preds, multioutput='raw_values')
            MAE = mean_absolute_error(seqs, preds, multioutput='raw_values')
            
            return {'r2': r2,
                    'MAE': MAE, 
                    'cosine_similarity_mean': np.mean(cos_sim, axis=0), 
                    'cosine_similarity_std': np.std(cos_sim, axis=0)
                   }
        
        elif mode in ['binary_classification', 'behavior_classification']:            
            preds = torch.cat([eval_data[i][0] for i in range(len(eval_data))]).ravel()
            labels = torch.cat([eval_data[i][1] for i in range(len(eval_data))]).ravel()
            session_idx = torch.cat([eval_data[i][2] for i in range(len(eval_data))]).ravel()

            cm = confusion_matrix(labels, preds, labels=kwargs.get('class_labels'))

            if kwargs.get('session_probs') and mode == 'binary_classification':
                probs = []
                unique_sessions = torch.unique(session_idx)

                for session in unique_sessions:
                    session_indices = torch.where(session_idx == session)[0]

                    preds_session = preds[session_indices]
                    labels_session = labels[session_indices]

                    probability = (preds_session == labels_session).float().mean().item()
                    probs.append(probability)
                
                names = []
                for path in self.loader.dataset.names:
                    names.append(os.path.splitext(os.path.basename(path))[0])

                sessions_probs = dict(zip(names, probs))
                return cm, sessions_probs

            return cm
            

    def test(self, 
             mode: str, 
             func_loss: Callable=None,  
             data_parallel: bool=False, 
             class_labels: Optional[torch.Tensor]=None, 
             class_weights: Optional[torch.Tensor]=None,
             session_probs: bool=False) -> tuple[Any, float]:
        '''
        Evaluate the model on the validation DataLoader.

        Args:
            mode (str): Evaluation mode.
            func_loss (Optional[Callable]): Loss function, can be None for inference only.
            data_parallel (bool): If True, use multiple GPUs.
            class_labels (Optional[torch.Tensor]): Required for classification tasks.
            class_weights (Optional[torch.Tensor]): Optional weights for loss calculation.
            session_probs (bool): If True, compute session-level probabilities.

        Returns:
            tuple:
                - stats: Dict for pretraining or (confusion_matrix, session_probs) for classification.
                - avg_loss: float, average loss over validation batches.
        '''
        if mode not in ['pretrain_reconstruction', 'binary_classification', 'behavior_classification']:
            raise KeyError('No such mode')
        if (mode == 'binary_classification' or mode == 'behavior_classification') and class_labels == None:
            raise KeyError('Need labels of classes for this mode')
            
        self._prepare_model(data_parallel)
        self.model.eval()
        
        eval_data = []
        eval_loss = []
        with torch.no_grad():
            for batch in tqdm(self.loader):
                batch_eval_data, loss_value = self._evaluate_batch(mode, 
                                                                   func_loss, 
                                                                   *batch, 
                                                                   class_labels=class_labels,
                                                                   class_weights=class_weights,
                                                                   session_probs=session_probs)
                eval_data.append(batch_eval_data)
                eval_loss.append(loss_value)
        
        stats = self._count_statistics(mode, eval_data, class_labels=class_labels, session_probs=session_probs)
        if self.save_path != None:
            self._save_loggs(mode, stats)

        return stats, np.mean(eval_loss)
    