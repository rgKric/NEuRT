import os
import json
import pandas as pd

class DataLoader:
    def __init__(self, cfg):
        """
        Initializes the DataLoader object.

        Parameters:
        ----------
        cfg : object
            Configuration object containing the attributes path, key_activity, and key_centroids.
        """
        self.path = cfg.path
        self.key_activity = cfg.key_activity
        self.key_centroids = cfg.key_centroids
        self.sep = cfg.sep
        self.header = cfg.header
        self.index_col = cfg.index_col

        # Load meta information
        self.meta = self._load_meta()

    def _load_meta(self):
        vpath = os.path.normpath(self.path)
        vlist = os.listdir(vpath)

        metafile = next(filter(lambda x: x.endswith('json'), vlist))
        with open(self.path + os.sep + metafile) as f:
            meta = json.load(f)
        return meta

    def __getitem__(self, index):
        if index < 0 or index >= len(self.meta):
            raise IndexError("Index out of range")

        m = self.meta[index]
        s = pd.read_csv(self.path + os.sep + m[self.key_activity[0]][self.key_activity[1]], sep=self.sep, header=self.header, index_col=self.index_col)
        c = pd.read_csv(self.path + os.sep + m[self.key_centroids[0]][self.key_centroids[1]], sep=self.sep, header=self.header, index_col=self.index_col)
        return (m, s, c)

    def __len__(self):
        return len(self.meta)