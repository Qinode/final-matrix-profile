import numpy as np


def sliding_dot_product(query, time_series):
    len_q, len_t = query.shape[0], time_series.shape[0]

    t_a = np.append(time_series, np.zeros((len_t, 1)))
    r_qa = np.append(query[::-1], np.zeros((2 * len_t - len_q, )))

    return np.fft.irfft(np.fft.rfft(r_qa) * np.fft.rfft(t_a))[len_q-1:len_t]


# compute the mean and standard deviation of all subsequence of a
# time series given a window size m
# could be optimized according to the paper:

# searching and mining trillions time series subsequences under
# dynamic time wrapping
def mean_std(window_size, time_series):
    len_t = time_series.shape[0]
    mean, std = np.zeros((len_t - window_size + 1)), np.zeros((len_t - window_size + 1))

    for i in range(len_t - window_size + 1):
        subsequence = time_series[i: i+window_size]
        mean[i] = np.mean(subsequence)
        std[i] = np.std(subsequence)

    return mean, std


def moving_average(time_series, window_size):
    cum_sum = np.cumsum(time_series, dtype=float)
    return (cum_sum[window_size-1:] - np.append(np.array([0]), cum_sum[:-window_size])) / window_size


def moving_std(time_series, moving_average, windows_size):
    cum_sum2 = np.cumsum(np.power(time_series, 2), dtype=float)
    cum_sum2 = cum_sum2[windows_size-1:] - np.append(np.array([0]), cum_sum2[:-windows_size])
    sigma2 = (cum_sum2 / windows_size) - np.power(moving_average, 2)
    return np.sqrt(sigma2)


def mass(query, time_series):
    m = query.shape[0]
    qt = sliding_dot_product(query, time_series)
    mean_q, std_q = np.mean(query), np.std(query)
    mean_t, std_t = mean_std(m, time_series)  # todo: optimize

    assert qt.shape == mean_t.shape == std_t.shape
    temp_term = (qt - m * mean_q * mean_t)/(m *std_q * std_t)
    d = np.sqrt(2 * m * (np.ones(qt.shape) - np.around(temp_term, decimals=9)))
    return d


def elementwise_min(mp, mpi, dp, idx):
    stack = np.stack((mp, dp))
    min_idx = np.where(np.argmin(stack, axis=0) == 1)
    mpi[min_idx] = idx
    return np.min(stack, axis=0), mpi


