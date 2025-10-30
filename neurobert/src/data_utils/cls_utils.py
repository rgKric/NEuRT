import os

from .datasets import Dataset4FineTune
from ..utils import load_data

def get_mice_sessions(folder: str,
                       class_label: int,
                         ignore_path: list[str]=[]) -> dict[int, tuple[list[str], int]]:
    '''
    Scan a folder for .npy session files and group them by mouse ID.

    Args:
        folder (str): Path to folder containing session .npy files.
        class_label (int): Label to assign to all sessions of each mouse.
        ignore_path (list[str], optional): List of file paths to ignore.

    Returns:
        dict[int, tuple[list[str], int]]: 
            Mapping from mouse ID to a tuple of (list of session file paths, class label).
    '''
    mouse_sessions = {}
    for session in os.listdir(folder):
        if not session.endswith('.npy'):
            continue
        parts = session.split("_") 
        mouse_id = int(parts[1])

        if mouse_id not in mouse_sessions:
            mouse_sessions[mouse_id] = [[], class_label]
        if os.path.join(folder, session) in ignore_path:
            continue
        mouse_sessions[mouse_id][0].append(os.path.join(folder, session))
    return mouse_sessions


def create_dataset(mouse_dict: dict, **kwargs) -> Dataset4FineTune:
    '''
    Create a Dataset4FineTune instance from a dictionary of mouse data.

    Args:
        mouse_dict (dict): Mapping from mouse ID (or key) to a tuple/list with:
            - paths to .npy files (list or str)
            - class label (int)
        **kwargs: Optional keyword arguments.
            - array_format (str): Either 'mem' (default) for memory-mapped arrays or 'arr' for full arrays.

    Returns:
        Dataset4FineTune: Dataset object containing concatenated data, shapes, labels, and names.
    '''
    data = []
    shapes = []
    labels = []
    names = []
    for _, val in mouse_dict.items():
        dataset, name = load_data(val[0], class_label=val[1], array_format=kwargs.get('array_format', 'mem'))
        data += dataset[0]
        shapes += dataset[1]
        labels += dataset[2]
        names += name

    return Dataset4FineTune(data, shapes, labels, names)