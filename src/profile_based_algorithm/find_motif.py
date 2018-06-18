import numpy as np
from src.util.util import moving_average
from src.util.util import moving_std
from src.util.util import mass


def find_motif(mp, mpi, window_size, time_series, time_series_freq, k=3, R=2):
    mp, mpi = mp.copy(), mpi.copy()
    result = []

    for i in range(k):
        min_dis, min_idx = np.min(mp), np.argmin(mp)
        min_dis = np.power(min_dis, 2)

        motif_pair_idx = np.sort(np.array([min_idx, mpi[min_idx]]))

        motif_idx = int(motif_pair_idx[0])
        query = time_series[motif_idx:motif_idx + window_size]

        mean_t = moving_average(time_series, window_size)
        std_t = moving_std(time_series, mean_t, window_size)

        distance_profile = mass(query, time_series_freq, time_series.shape[0], mean_t, std_t, mean_t[motif_idx], std_t[motif_idx])
        distance_profile = np.power(distance_profile, 2)

        exclude_window = int(np.around(np.divide(window_size, 2)))

        exclude_start = max(0, motif_idx - exclude_window)
        exclude_end = min(motif_idx + exclude_window + 1, mp.shape[0])

        distance_profile[exclude_start:exclude_end] = np.inf
        distance_profile[distance_profile > (min_dis * R)] = np.inf

        motif_idx = int(motif_pair_idx[1])
        exclude_start = max(0, motif_idx - exclude_window)
        exclude_end = min(motif_idx + exclude_window + 1, mp.shape[0])
        distance_profile[exclude_start:exclude_end] = np.inf

        sorted_dp, sorted_dp_idx = np.sort(distance_profile), np.argsort(distance_profile)
        neighbor_idx = -1 * np.ones((10, 1))

        for i in range(10):
            if sorted_dp[i] == np.inf or sorted_dp.shape[0] < i:
                break
            else:
                neighbor_idx[i] = sorted_dp_idx[0]
                sorted_dp, sorted_dp_idx = np.delete(sorted_dp, 0), np.delete(sorted_dp_idx, 0)

                selected_neighbor = np.where(np.abs(sorted_dp_idx - neighbor_idx[i]) < exclude_window)
                sorted_dp, sorted_dp_idx = np.delete(sorted_dp, selected_neighbor), np.delete(sorted_dp_idx, selected_neighbor)

        neighbor_idx = np.delete(neighbor_idx, np.where(neighbor_idx == -1)[0])

        result.append((motif_pair_idx, neighbor_idx))

        all_idx = np.append(motif_pair_idx, neighbor_idx)

        for idx in all_idx:
            exclude_start = int(max(0, idx - exclude_window))
            exclude_end = int(min(idx + exclude_window + 1, mp.shape[0]))
            mp[exclude_start: exclude_end] = np.inf

    return result
