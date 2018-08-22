import numpy as np
import logging
from src.distance.z_norm_euclidean import z_norm_euclidean
from src.util.util import mass, elementwise_min, moving_average, moving_std, sliding_dot_product, distance_profile

logger = logging.getLogger(__name__)

# def marix_profile(series1, series2, window, self_join, method='naive', distance='z-norm'):
#     return stamp(series1, series2, window, self_join, distance)
    # return naive(series1, series2, window, distance)


def naive(series1, series2, window, distance=None):
    self_join = np.allclose(series1, series2)

    dm = np.full((series1.shape[0] - window + 1, series2.shape[0] - window + 1), np.inf)
    for i in range(series1.shape[0] - window + 1):
        for j in range(series2.shape[0] - window + 1):
            if self_join:
                if i - (window/2) <= j <= i + (window/2):
                    continue
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

        if self_join:
            dp = mass(series2[i:i + window_size], series1_freq, series1.shape[0], mean_t, std_t, mean_t[i], std_t[i])
            left, right = max(0, i - window_size//2), min((i + window_size//2) + 1, n2)
            dp[left:right] = np.inf
        else:
            query = series2[i:i + window_size]
            mean_q = np.mean(query)
            std_q = np.mean(query)
            dp = mass(query, series1_freq, series1.shape[0], mean_t, std_t, mean_q, std_q)

        p_ab, i_ab = elementwise_min(p_ab, i_ab, dp, i)

    return p_ab, i_ab


# series1 join series2
def stomp(series1, series2, window_size, self_join, distance=None):

    n2 = series2.shape[0]

    mean_t = moving_average(series1, window_size)
    std_t = moving_std(series1, mean_t, window_size)

    series1_freq = np.fft.fft(np.append(series1, np.zeros(window_size, )))
    qt = sliding_dot_product(series2[0: 0 + window_size], series1_freq, series1.shape[0])
    qt_1 = qt.copy()

    dp = distance_profile(qt, window_size, mean_t, std_t, np.mean(series2[0: 0 + window_size]), np.std(series2[0: 0 + window_size]))

    if self_join:
        left, right = max(0, 0 - window_size//2), min((0 + window_size//2) + 1, n2)
        dp[left: right] = np.inf

    mp, mpi = dp.copy(), np.zeros(dp.shape[0])

    for i in range(1, n2 - window_size + 1):
        print('Process: {}/{}'.format(i, n2-window_size+1))
        # for j in reversed(range(1, n2 - window_size + 1)):
        #     qt[j] = qt[j - 1] - (series1[i - 1] * series1[j - 1]) + (series1[i + window_size - 1] * series1[j + window_size - 1])

        qt = np.roll(qt, 1)
        term1 = series1[i - 1] * np.roll(series1[: n2 - window_size + 1], 1).reshape(qt.shape)
        term2 = series1[i + window_size - 1] * series1[window_size - 1:].reshape(qt.shape)
        qt = qt - term1 + term2
        qt[0] = qt_1[i]

        if self_join:
            dp = distance_profile(qt, window_size, mean_t, std_t, mean_t[i], std_t[i])
            left, right = max(0, i - window_size//2), min((i + window_size//2) + 1, n2)
            dp[left:right] = np.inf
        else:
            query = series2[i:i + window_size]
            mean_q = np.mean(query)
            std_q = np.mean(query)
            dp = mass(qt, window_size, mean_t, std_t, mean_q, std_q)

        mp, mpi = elementwise_min(mp, mpi, dp, i)

    return mp, mpi

# series1 join series2
def lrstomp(series1, series2, window_size, self_join, distance=None):

    n2 = int(series2.shape[0])

    mean_t = moving_average(series1, window_size)
    std_t = moving_std(series1, mean_t, window_size)

    series1_freq = np.fft.fft(np.append(series1, np.zeros(window_size, )))
    qt = sliding_dot_product(series2[0: 0 + window_size], series1_freq, series1.shape[0])
    qt_1 = qt.copy()

    dp = distance_profile(qt, window_size, mean_t, std_t, np.mean(series2[0: 0 + window_size]), np.std(series2[0: 0 + window_size]))

    if self_join:
        left, right = max(0, 0 - window_size//2), min((0 + window_size//2) + 1, n2)
        dp[left: right] = np.inf

    mp, mpi = dp.copy(), np.zeros(dp.shape[0])

    mp_left, mp_right = np.full((n2 - window_size + 1, ), np.inf), np.full((n2 - window_size + 1, ), np.inf)
    mpi_left, mpi_right = -1 * np.ones((n2 - window_size + 1, )), -1 * np.ones((n2 - window_size + 1, ))

    mp_left[1:], mpi_left[1:] = elementwise_min(mp_left[1:], mpi_left[1:], dp[1:], 0)

    for i in range(1, n2 - window_size + 1):
        print('{}/{}'.format(i + 1, n2 - window_size + 1))

        # for j in reversed(range(1, n2 - window_size + 1)):
        #     qt[j] = qt[j - 1] - (series1[i - 1] * series1[j - 1]) + (series1[i + window_size - 1] * series1[j + window_size - 1])

        qt = np.roll(qt, 1)
        term1 = series1[i - 1] * np.roll(series1[: n2 - window_size + 1], 1).reshape(qt.shape)
        term2 = series1[i + window_size - 1] * series1[window_size - 1:].reshape(qt.shape)
        qt = qt - term1 + term2
        qt[0] = qt_1[i]

        if self_join:
            dp = distance_profile(qt, window_size, mean_t, std_t, mean_t[i], std_t[i])
            left, right = max(0, i - window_size//2), min((i + window_size//2) + 1, n2)
            dp[left:right] = np.inf
        else:
            query = series2[i:i + window_size]
            mean_q = np.mean(query)
            std_q = np.mean(query)
            dp = mass(qt, window_size, mean_t, std_t, mean_q, std_q)

        mp, mpi = elementwise_min(mp, mpi, dp, i)
        mp_left[i+1:], mpi_left[i+1:] = elementwise_min(mp_left[i+1:], mpi_left[i+1:], dp[i+1:], i)
        mp_right[:i], mpi_right[:i] = elementwise_min(mp_right[:i], mpi_right[:i], dp[:i], i)

    return (mp, mpi), (mp_left, mpi_left), (mp_right, mpi_right)


# def stampi(series1, series2, new1, new2, p, i, window_size, self_join, distance=None):
#     if series1.shape[0] < window_size or series2.shape[0]:
#         pass
#     else:
#         series2_new = np.append(series2, new2)
#         series1_new = np.append(series1, new1)
#
#         if self_join:
#             query = series2_new[-window_size:]
#             idx = series2.shape[0] + 1 - window_size
#             dp = mass(query, series1)
#
#             left, right = max(0, i - window_size//2), min(i + window_size//2, dp.shape[0])
#             dp[left:right+1] = np.inf
#
#             p, i = elementwise_min(p, i, dp, idx)
#             p_new, i_new = np.min(dp), np.argmin(dp)
#
#             return np.append(p, p_new), np.append(i, i_new)
#         else:
#             query = series2_new[-window_size:]
#             idx = series2.shape[0] + 1 - window_size
#             dp1 = mass(query, series1)
#
#             p, i = elementwise_min(p, i, dp1, idx)
#             last_sequence = series1_new[-window_size:]
#             dp2 = mass(last_sequence, series2_new)
#             p_new, i_new = np.min(dp2), np.argmin(dp2)
#
#             return np.append(p, p_new), np.append(i, i_new)

