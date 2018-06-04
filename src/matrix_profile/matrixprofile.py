import numpy as np
from src.distance.z_norm_euclidean import z_norm_euclidean


def marix_profile(series1, series2, window, method='naive', distance='z-norm'):
    return naive(series1, series2, window, distance)


def naive(series1, series2, window, distance):
    dm = np.full((series1.shape[0] - window + 1, series2.shape[0] - window + 1), np.inf)
    for i in range(series1.shape[0] - window + 1):
        for j in range(series2.shape[0] - window + 1):
            if i - (window/2) <= j <= i + (window/2):
                continue
            else:
                dm[i][j] = z_norm_euclidean(series2[j: j + window], series1[i: i + window])

    return np.amin(dm, axis=0).reshape(series1.shape[0] - window + 1, 1), np.argmin(dm, axis=0).reshape(series1.shape[0] - window + 1, 1)


def stamp(series1, series2, window, distance):
    pass
