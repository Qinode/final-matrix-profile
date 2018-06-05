import numpy as np
from src.distance.z_norm_euclidean import z_norm_euclidean
from src.util.util import mass, elementwise_min


# def marix_profile(series1, series2, window, self_join, method='naive', distance='z-norm'):
#     return stamp(series1, series2, window, self_join, distance)
    # return naive(series1, series2, window, distance)


def naive(series1, series2, window, distance=None):
    dm = np.full((series1.shape[0] - window + 1, series2.shape[0] - window + 1), np.inf)
    for i in range(series1.shape[0] - window + 1):
        for j in range(series2.shape[0] - window + 1):
            if i - (window/2) <= j <= i + (window/2):
                continue
            else:
                dm[i][j] = z_norm_euclidean(series2[j: j + window], series1[i: i + window])

    return np.amin(dm, axis=0).reshape(series1.shape[0] - window + 1, 1), np.argmin(dm, axis=0).reshape(series1.shape[0] - window + 1, 1)


def stamp(series1, series2, window_size, self_join, distance=None):
    n_b = series2.shape[0]
    p_ab = np.full((series1.shape[0] - window_size + 1,), np.inf)
    i_ab = np.zeros((series1.shape[0] - window_size + 1,))

    for i in range(n_b - window_size + 1):
        dp = mass(series1[i:i + window_size], series1)

        if self_join:
            left, right = max(0, i - window_size//2), min(i + window_size//2, n_b)
            dp[left:right+1] = np.inf

        p_ab, i_ab = elementwise_min(p_ab, i_ab, dp, i)

    return p_ab, i_ab
