import numpy as np
from scipy.signal import butter, lfilter, filtfilt

def normSignals(signals):
    """
    Normalizes signals in a pandas.DataFrame.

    Parameters:
    ----------
    signals : pandas.DataFrame
        Input signals to be normalized.

    Returns:
    -------
    pandas.DataFrame
        Normalized signals.

    Notes:
    ----------
    Normalization is performed using the formula:
    (signals - min) / (max - min)
    where min and max are the minimum and maximum values of the input signals respectively.
    """
    min = signals.min()
    max = signals.max()
    return (signals - min) / (max - min)



def filt_butter(x, freq: float, btype: str, fs):
    """
    Filter 1d timeseries with Butterworth filter using
    :func:`scipy.signal.butter`.

    Parameters
    ----------
    x : np.ndarray
        Input timeseries.
    freq : float
        Cut-off frequency.
    btype : str
        Either "low" or "high" specify low or high pass filtering.

    Returns
    -------
    x_filt : np.ndarray
        Filtered timeseries.
    """
    but_b, but_a = butter(2, freq * 2, btype=btype, analog=False, fs=fs)
    return filtfilt(but_b, but_a, x)



class FilterNeighbors:
    def __init__(self, distance, cfg):
        self.distance = distance
        self.unit_id_key = cfg.id_col
        self.x_key = cfg.x_col
        self.y_key = cfg.y_col

    def __call__(self, df, unit_id):
        # Find the coordinates of the unit with the given unit_id
        point = df[df[self.unit_id_key] == unit_id].iloc[0][[self.x_key, self.y_key]].values
        # Extract the coordinates of the unit
        x, y = point
        # Calculate the squared distance from the given unit to all units in the DataFrame
        df['distance'] = (df[self.x_key] - x)**2 + (df[self.y_key] - y)**2
        # Exclude the unit itself from consideration
        df_filtered = df[df[self.unit_id_key] != unit_id]
        # Find units that are within the specified distance
        less_than_distance = df_filtered[df_filtered['distance'] < self.distance * self.distance][self.unit_id_key].tolist()
        # Find units that are beyond the specified distance
        greater_than_distance = df_filtered[df_filtered['distance'] > self.distance * self.distance][self.unit_id_key].tolist()
        return less_than_distance, greater_than_distance

    

def prepair_vector(signal_unit_id, centroid_unit_id, signals, centroids, filter_neighbors):
    s = np.array(signals[signal_unit_id])
    less_than_distance, greater_than_distance = filter_neighbors(centroids, centroid_unit_id)
    if len(less_than_distance) < 2 or len(greater_than_distance) < 2:
        return None
    m_i_1 = np.array(signals[[str(col) for col in less_than_distance]].mean(axis=1))
    sd_i_1 = np.array(signals[[str(col) for col in less_than_distance]].std(axis=1))
    m_i_2 = np.array(signals[[str(col) for col in greater_than_distance]].mean(axis=1))
    sd_i_2 = np.array(signals[[str(col) for col in greater_than_distance]].std(axis=1))
    return np.vstack((s, m_i_1, sd_i_1, m_i_2, sd_i_2))



def prepair_vector_with_weight(signal_unit_id, centroid_unit_id, signals, centroids, filter_neighbors):
    s = np.array(signals[signal_unit_id])
    less_than_distance, greater_than_distance = filter_neighbors(centroids, centroid_unit_id)
    if len(less_than_distance) < 2 or len(greater_than_distance) < 2:
        return None
    m_i_1 = np.array(signals[[str(col) for col in less_than_distance]].mean(axis=1))
    sd_i_1 = np.array(signals[[str(col) for col in less_than_distance]].std(axis=1))
    m_i_2 = np.array(signals[[str(col) for col in greater_than_distance]].mean(axis=1))
    sd_i_2 = np.array(signals[[str(col) for col in greater_than_distance]].std(axis=1))
    return np.vstack((s, m_i_1, sd_i_1, m_i_2, sd_i_2))