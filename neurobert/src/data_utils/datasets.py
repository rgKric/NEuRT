import numpy as np

import torch
from torch.utils.data import Dataset
from torch import from_numpy

from sklearn.utils.class_weight import compute_class_weight

from typing import Namedtuple


class Preprocess4Mask:
    ''' Pre-processing steps for pretraining transformer '''
    def __init__(self, mask_cfg: Namedtuple):
        '''
        Initializes preprocessing parameters from a configuration Namedtuple.

        Args:
            mask_cfg (Namedtuple): Configuration containing:
                - mask_ratio (float): Fraction of tokens to mask.
                - max_gram (int): Maximum span length to mask.
                - mask_prob (float): Probability to zero out the masked span.
                - replace_prob (float): Probability to replace masked span with random values.
        '''
        self.mask_ratio = mask_cfg.mask_ratio
        self.max_gram = mask_cfg.max_gram
        self.mask_prob = mask_cfg.mask_prob
        self.replace_prob = mask_cfg.replace_prob
    

    def span_mask(self, 
                  seq_len: int,
                  max_gram: int=3, 
                  p: float=0.2,
                  goal_num_predict: int=15) -> list[int]:
        '''
        Generates masked positions in a sequence using span masking.

        Args:
            seq_len (int): Length of the sequence.
            max_gram (int): Maximum span length for masking.
            p (float): Base probability for span lengths.
            goal_num_predict (int): Target number of masked positions.

        Returns:
            list[int]: list of positions to mask in the sequence.
        '''
        ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
        pvals = p * np.power(1 - p, np.arange(max_gram))
        pvals /= pvals.sum(keepdims=True)
        mask_pos = set()
        while len(mask_pos) < goal_num_predict:
            n = np.random.choice(ngrams, p=pvals)
            n = min(n, goal_num_predict - len(mask_pos))
            anchor = np.random.randint(seq_len)
            if anchor in mask_pos:
                continue
            for i in range(anchor, min(anchor + n, seq_len - 1)):
                mask_pos.add(i)
        return list(mask_pos)


    def __call__(self, instance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Applies span masking to a single instance.

        Args:
            instance (np.ndarray): Input array of shape (seq_len, feature_dim).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - instance_mask: Array with masked values replaced.
                - mask_pos_index: Positions of the masked tokens.
                - seq: Original values at masked positions.
        '''
        shape = instance.shape
        n_pred = max(1, int(round(shape[0] * self.mask_ratio)))
        mask_pos = self.span_mask(shape[0], self.max_gram,  goal_num_predict=n_pred)
        instance_mask = instance.copy()

        mask_pos_index = mask_pos
        if np.random.rand() < self.mask_prob:
            instance_mask[mask_pos, :] = np.zeros((len(mask_pos), shape[1]))
        else:
            instance_mask[mask_pos, :] = np.random.random((len(mask_pos), shape[1]))
        seq = instance[mask_pos_index, :]
        return instance_mask, np.array(mask_pos_index), np.array(seq)
    

class DatasetBase(Dataset):
    def __init__(self, data, shapes, pipeline=[]):
        super().__init__()
        self.data = data
        self.shapes = shapes
        self.cumulative_sizes = np.cumsum([0] + [shape[0] for shape in shapes])
    
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def _get_idx(self, idx):
        file_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
        in_file_idx = idx - self.cumulative_sizes[file_idx]
        return file_idx, in_file_idx
    

class Dataset4Pretrain(DatasetBase):
    def __init__(self, data, shapes, pipeline=[]):
        super().__init__(data, shapes)
        self.pipeline = pipeline

    def __getitem__(self, idx):
        file_idx, in_file_idx = self._get_idx(idx)
        instance = np.array(self.data[file_idx][in_file_idx])
        for proc in self.pipeline:
            instance = proc(instance)
        mask_seq, masked_pos, seq = instance
        return from_numpy(mask_seq).type(torch.float32), from_numpy(masked_pos).long(), from_numpy(seq).type(torch.float32)
    

class Dataset4FineTune(DatasetBase):
    def __init__(self, data, shapes, labels, names):
        super().__init__(data, shapes)
        self.labels = labels
        self.names = names

    def __getitem__(self, idx):
        file_idx, in_file_idx = self._get_idx(idx)
        label = self.labels[file_idx][in_file_idx]
        instance = np.array(self.data[file_idx][in_file_idx])
        return from_numpy(instance).type(torch.float32), label, file_idx
    
    def count_weigths(self):
        '''
        Compute class weights, but we did not use it
        '''
        label_list = np.concatenate([y for y in self.labels], axis=0)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(label_list), y=label_list)
        weights = torch.tensor(class_weights, dtype=torch.float32)
        return weights
    
