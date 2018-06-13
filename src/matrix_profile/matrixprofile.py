import numpy as np
import logging
from src.distance.z_norm_euclidean import z_norm_euclidean
from src.util.util import mass, elementwise_min, moving_average, moving_std

logger = logging.getLogger(__name__)

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


# series1 join series2
def stamp(series1, series2, window_size, self_join, distance=None):
    n2 = series2.shape[0]
    p_ab = np.full((series1.shape[0] - window_size + 1,), np.inf)
    i_ab = np.zeros((series1.shape[0] - window_size + 1,))

    mean_t = moving_average(series1, window_size)
    std_t = moving_std(series1, mean_t, window_size)

    series1_freq = np.fft.fft(np.append(series1, np.zeros(window_size, )))

    for i in range(n2 - window_size + 1):
        print('{}/{}'.format(i + 1, n2 - window_size + 1))
        dp = mass(series2[i:i + window_size], series1_freq, series1.shape[0], mean_t, std_t)

        if self_join:
            left, right = max(0, i - window_size//2), min(i + window_size//2, n2)
            dp[left:right+1] = np.inf

        p_ab, i_ab = elementwise_min(p_ab, i_ab, dp, i)

    return p_ab, i_ab


def stampi(series1, series2, new1, new2, p, i, window_size, self_join, distance=None):
    if series1.shape[0] < window_size or series2.shape[0]:
        pass
    else:
        series2_new = np.append(series2, new2)
        series1_new = np.append(series1, new1)

        if self_join:
            query = series2_new[-window_size:]
            idx = series2.shape[0] + 1 - window_size
            dp = mass(query, series1)

            left, right = max(0, i - window_size//2), min(i + window_size//2, dp.shape[0])
            dp[left:right+1] = np.inf

            p, i = elementwise_min(p, i, dp, idx)
            p_new, i_new = np.min(dp), np.argmin(dp)

            return np.append(p, p_new), np.append(i, i_new)
        else:
            query = series2_new[-window_size:]
            idx = series2.shape[0] + 1 - window_size
            dp1 = mass(query, series1)

            p, i = elementwise_min(p, i, dp1, idx)
            last_sequence = series1_new[-window_size:]
            dp2 = mass(last_sequence, series2_new)
            p_new, i_new = np.min(dp2), np.argmin(dp2)

            return np.append(p, p_new), np.append(i, i_new)

