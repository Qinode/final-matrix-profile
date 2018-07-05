import numpy as np


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