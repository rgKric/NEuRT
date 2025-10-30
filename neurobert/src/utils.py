import os
import json

import numpy as np
import torch
from torch.utils.data import random_split
import random

from typing import Union, Optional, Any
from torch.utils.data import Dataset, Subset


def load_data(path: Union[str, list[str]], 
              ignore_paths: Optional[list[str]]=[], 
              class_label: int=None,
              array_format: str='mem'):
    """
    Loads .npy data files from a directory or list of file paths and optionally generates class labels.

    Args:
        path (Union[str, list[str]]): Directory path containing .npy files or a list of .npy file paths.
        ignore_paths (Optional[list[str]]): List of file paths to ignore. Defaults to empty list.
        class_label (Optional[int]): If provided, generates labels (0 or 1) for each loaded array. Defaults to None.
        array_format (str): 'mem' to use memory-mapped arrays, 'arr' to load full arrays into memory. Defaults to 'mem'.

    Returns:
        tuple:
            - tuple of:
                - data (list[np.ndarray]): List of loaded arrays.
                - shapes (list[tuple[int, ...]]): Shapes of each array.
                - labels (Optional[list[np.ndarray]]): List of label arrays if class_label is provided, else None.
            - npy_files (list[str]): List of file paths that were actually loaded.

    Raises:
        TypeError: If `path` is neither a string nor a list of strings.
    """
    if isinstance(path, list):
        npy_files = path
    elif isinstance(path, str):
        npy_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.npy')]
    else:
        raise TypeError('Wrong type of path, can be list() or str()')
        
    shapes = []
    data = []
    for file_path  in npy_files:
        if file_path not in ignore_paths:
            if array_format == 'mem':
                data_array = np.lib.format.open_memmap(file_path, mode='r')
            elif array_format == 'arr':
                data_array = np.load(file_path)
            shape = data_array.shape
            data.append(data_array)
            shapes.append(shape)
            
    if class_label != None:
        labels = []
        for shape in shapes:
            if class_label == 0:
                label = np.zeros((shape[0],), dtype=np.int64)
            else:
                label = np.ones((shape[0],), dtype=np.int64)
            labels.append(label)
        return (data, shapes, labels), npy_files
    
    return (data, shapes), npy_files


def split_dataset(dataset: Dataset, train_part: float, seed: int=42) -> tuple[Subset, Subset]:
    """
    Splits a PyTorch dataset into training and test subsets.

    Args:
        dataset (Dataset): PyTorch dataset to split.
        train_part (float): Fraction of the dataset to use for training (between 0 and 1).
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[Subset, Subset]: Training and test dataset subsets.

    Notes:
        This uses PyTorch's `random_split` internally.
    """
    generator = torch.Generator().manual_seed(seed)
    train_size = int(train_part * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    
    return train_dataset, test_dataset


def get_device(gpu: str=None) -> torch.device:
    """
    Returns a PyTorch device (CPU or GPU) based on availability and user preference.

    Args:
        gpu (Optional[str]): GPU index as a string (e.g., '0', '1'). If None, automatically selects GPU 0 if available. Defaults to None.

    Returns:
        torch.device: Selected device.
    """
    if gpu is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device


def set_seed(seed: int) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch (CPU and GPU) to ensure reproducibility.

    Args:
        seed (int): Seed value to use for all random number generators.

    Returns:
        None
    """
    random.seed(seed)                        # Python
    np.random.seed(seed)                     # NumPy
    torch.manual_seed(seed)                  # CPU
    torch.cuda.manual_seed(seed)             # CUDA
    torch.cuda.manual_seed_all(seed)         # multiple GPUs


def validate_and_prepare_start_config(config_path: str, model_mode: str, script_mode: str = None) -> dict[str, Any]:
    """
    Loads a JSON configuration file, validates required keys, normalizes all paths,
    and ensures that required files/folders exist.

    Args:
        config_path (str): Path to the JSON configuration file.
        model_mode (str): 'train' or 'test' — affects path_to_model validation.
        script_mode (str): 'recon' or 'cls' — determines config structure.

    Returns:
        dict[str, Any]: Loaded and validated configuration.
    """

    with open(config_path, 'r') as f:
        config = json.load(f)

    if script_mode == 'recon':
        required_keys = [
            'path_to_data',
            'path_to_model',
            'path_to_model_cfg',
            'path_to_trainer_cfg',
            'save_path',
        ]
    elif script_mode == 'cls':
        required_keys = [
            'path_to_WT',
            'path_to_5xFAD',
            'path_to_model',
            'path_to_model_cfg',
            'path_to_trainer_cfg',
            'save_path',
            'train_indexes',
            'val_indexes',
            'test_indexes',
        ]
    else:
        raise ValueError(f"Unknown script_mode: {script_mode}")

    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing key in config: '{key}'")

    save_path = os.path.normpath(config['save_path'])
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print(f"Created missing folder: '{save_path}'")
    elif not os.path.isdir(save_path):
        raise NotADirectoryError(f"'{save_path}' exists but is not a directory")
    config['save_path'] = save_path

    def check_path(key: str, expect_dir=False, allow_null=False):
        value = config.get(key)
        if value is None:
            if allow_null:
                return
            raise ValueError(f"'{key}' cannot be null")
        value = os.path.normpath(value)
        config[key] = value
        if not os.path.exists(value):
            raise FileNotFoundError(f"Path for '{key}' does not exist: {value}")
        if expect_dir and not os.path.isdir(value):
            raise NotADirectoryError(f"'{key}' must be a directory: {value}")

    if script_mode == 'recon':
        check_path('path_to_data', expect_dir=True)
        check_path('path_to_model_cfg')
        check_path('path_to_trainer_cfg')
        check_path('path_to_model', allow_null=(model_mode == 'test'))

        if 'path_to_mask_cfg' in config:
            check_path('path_to_mask_cfg', allow_null=True)

    elif script_mode == 'cls':
        check_path('path_to_WT', expect_dir=True)
        check_path('path_to_5xFAD', expect_dir=True)
        check_path('path_to_model_cfg')
        check_path('path_to_trainer_cfg')
        check_path('path_to_model', allow_null=(model_mode == 'test'))

        for k in ['train_indexes', 'val_indexes', 'test_indexes']:
            v = config.get(k)
            if v == []:
                config[k] = None
            elif v is not None and not isinstance(v, list):
                raise TypeError(f"'{k}' must be a list or null")

    return config