import numpy as np


def sliding_dot_product(query, time_series_freq, len_t):
    len_q = query.shape[0]

    r_qa = np.append(query[::-1], np.zeros((len_t, )))

    return np.fft.ifft(np.fft.fft(r_qa) * time_series_freq)[len_q - 1:len_t]


def moving_average(time_series, window_size):
    cum_sum = np.nancumsum(time_series, dtype=float)
    return (cum_sum[window_size-1:] - np.append(np.array([0]), cum_sum[:-window_size])) / window_size


def moving_std(time_series, moving_average, windows_size):
    cum_sum2 = np.nancumsum(np.power(time_series, 2), dtype=float)
    cum_sum2 = cum_sum2[windows_size-1:] - np.append(np.array([0]), cum_sum2[:-windows_size])
    sigma2 = (cum_sum2 / windows_size) - np.power(moving_average, 2)
    return np.sqrt(sigma2)


def distance_profile(qt, window_size, mean_t, std_t, mean_q, std_q):

    std_term = std_q * std_t
    zeros_pos = np.where(std_term == 0)
    std_term[zeros_pos] = 1

    temp_term = (qt - window_size * mean_q * mean_t)/std_term

    d = 2 * (window_size * np.ones(temp_term.shape) - temp_term)
    return np.sqrt(np.abs(d))


def mass(query, time_series_freq, len_t, mean_t, std_t, mean_q, std_q):
    m = query.shape[0]
    qt = sliding_dot_product(query, time_series_freq, len_t)

    temp_term = (qt - m * mean_q * mean_t)/(std_q * std_t)
    d = 2 * (m * np.ones(temp_term.shape) - temp_term)
    return np.sqrt(np.abs(d))


def elementwise_min(mp, mpi, dp, idx):
    min_idx = dp < mp

    mpi[min_idx] = idx
    mp[min_idx] = dp[min_idx]
    
    return np.minimum(mp, dp), mpi
    # mp[idx], mpi[idx] = np.min(dp), np.argmin(dp)
    # return mp, mpi


