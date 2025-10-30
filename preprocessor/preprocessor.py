import numpy as np
from utils import normSignals, prepair_vector

class Preprocessor:
    def __init__(self, cfg):
        self.is_norm = cfg.is_norm
        self.delta = cfg.delta
        self.length = cfg.length
        self.number = cfg.number
        
    def __call__(self, dataloader, filter, path):
        for meta, signals, centroids in dataloader:
            dataset = []
            print(meta)
            if self.is_norm:
                meta, signals, centroids = (meta, normSignals(signals), centroids)
            for signal_unit_id, centroid_unit_id in zip(signals.columns, centroids['unit_id']):
                vector = prepair_vector(signal_unit_id, centroid_unit_id, signals, centroids, filter)
                if vector is not None:
                    for start in np.random.randint(0, signals.shape[0] - self.length, self.number):
                        dataset.append(vector[:,start:start + self.length].T)
            dataset = np.array(dataset)
            print(dataset.shape)
            np.save(path + '/session_' + str(meta['session_idx']) + '_scan_' + str(meta['scan_idx']) + '_field_' + str(meta['field_idx']), dataset)