import numpy as np
from src.mds.cmds import cmdscale
import scipy.spatial.distance


def standarization(data):
    """

    :param data:
        n x window_size
    :return:
        standarization data
    """

    for i in range(data.shape[0]):
        mean = np.mean(data[i])
        std = np.std(data[i])
        if std != 0:
            data[i] = (data[i] - mean)/std
        else:
            data[i] = data[i] - mean

    return data


def pattern_mds(idx, window_size, time_series):
    data = np.zeros((len(idx), window_size))

    for i, value in enumerate(idx):
        data[i] = time_series[value: value+window_size].reshape(data[i].shape)

    z_data = standarization(data)
    pd = scipy.spatial.distance.pdist(z_data)
    pdm = scipy.spatial.distance.squareform(pd)
    return cmdscale(pdm)