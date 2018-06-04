import numpy as np


def sliding_dot_product(query, time_series):
    len_q, len_t = query.shape[0], time_series.shape[0]

    t_a = np.append(time_series, np.zeros((len_t, 1)))
    r_qa = np.append(query[::-1], np.zeros((2 * len_t - len_q, )))

    r_qaf, t_af = np.fft.rfft(r_qa), np.fft.rfft(t_a)

    return np.fft.irfft(r_qaf * t_af)[len_q-1:len_t]


# compute the mean and standard deviation of all subsequence of a
# time series given a window size m
# could be optimized according to the paper:

# searching and mining trillions time series subsequences under
# dynamic time wrapping
def mean_std(window_size, time_series):
    len_t = time_series.shape[0]
    mean = std = np.zeros((len_t - window_size + 1))

    for i in range(len_t - window_size + 1):
        subsequence = time_series[i: i+window_size]
        mean[i] = np.mean(subsequence)
        std[i] = np.std(subsequence)

    return mean, std


def mass(query, time_series):
    m = query.shape[0]
    qt = sliding_dot_product(query, time_series)
    mean_q, std_q = np.mean(query), np.std(query)
    mean_t, std_t = mean_std(m, time_series)

    assert qt.shape == mean_t.shape == std_t.shape
    d1 = (qt[1] - m * mean_q * mean_t[1]/std_q * std_t[1])
    d = np.sqrt(2 * m * (np.ones(qt.shape) - np.divide(qt - m * mean_q * mean_t, m * std_q * std_t)))
    return d


