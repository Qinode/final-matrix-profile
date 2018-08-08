import os

import numpy as np
import scipy.io as sio

from src.util.util import moving_std, moving_average, sliding_dot_product, distance_profile
from paper.MPIII.util import discretization_pre
from paper.MPIII.virtual_hypothesis.Cluster import Cluster


def ts_cluster(ts, window_size, mp, mpi, bits):
    ts, mp, mpi = ts.copy(), mp.copy(), mpi.copy()
    t_min, t_max = discretization_pre(ts, window_size)
    clusters = []

    ts_freq = np.fft.fft(np.append(ts, np.zeros((window_size, ))))
    ma = moving_average(ts, window_size)
    mstd = moving_std(ts, ma, window_size)

    max_pattern_size = ts.shape[0] // window_size
    patterns = []
    pattern_size = 0

    # create new cluster
    while pattern_size <= max_pattern_size:
        bit_save = []
        temp_clusters = []
        exc_zone = []

        idx1 = np.argmin(mp)
        idx2 = mpi[idx1]

        sub1 = ts[idx1: idx1+window_size]
        sub2 = ts[idx2: idx2+window_size]

        new_cluster = Cluster(2, sub1+sub2/2.0, np.c_[sub1, sub2], np.array([idx1, idx2]))
        bit_save.append(new_cluster.description_length(t_min, t_max, bits))
        temp_clusters.append(clusters + [new_cluster])
        exc_zone.append([idx1, idx2])

        # add pattern to an exist cluster
        for a_cluster_idx in range(len(clusters)):
            center = clusters[a_cluster_idx].center
            qt = sliding_dot_product(center, ts_freq, ts.shape[0])
            dp = distance_profile(qt, window_size, ma, mstd, np.mean(center), np.std(center))

            for i in patterns:
                start = min(0, i - (window_size//2))
                end = max(i + (window_size//2), dp.shape[0])
                dp[start: end] = np.inf

            nn_idx = np.argmin(dp)
            temp_new_cluster = Cluster.add_pattern(clusters[a_cluster_idx], ts[nn_idx: nn_idx+window_size], nn_idx)
            bit_save.append(temp_new_cluster.description_length(t_min, t_max, bits))
            temp_clusters.append(clusters[:a_cluster_idx] + [temp_new_cluster] + clusters[a_cluster_idx+1:])
            exc_zone.append([nn_idx])

        # merge two cluster
        for a_cluster_idx in range(len(clusters)):
            for b_cluster_idx in range(a_cluster_idx, len(clusters)):
                if a_cluster_idx != b_cluster_idx:
                    merged_cluster = Cluster.merge_cluster(clusters[a_cluster_idx], clusters[b_cluster_idx])
                    bit_save.append(merged_cluster.description_length(t_min, t_max, bits))
                    temp_clusters.append(clusters[:a_cluster_idx] + clusters[a_cluster_idx+1:b_cluster_idx] + clusters[b_cluster_idx+1:] + [merged_cluster])
                    exc_zone.append([])

        bs, idx = np.max(np.array(bit_save)), np.argmax(np.array(bit_save))

        if bs < 0:
            break
        else:
            clusters = temp_clusters[idx]
            exc_idx = exc_zone[idx]
            for i in exc_idx:
                pattern_size += 1
                patterns.append(i)

                start = max(0, i - (window_size//2))
                end = min(i + (window_size//2), mp.shape[0])
                mp[start:end] = np.inf
                ts[start:end] = np.inf

    return clusters


if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    eval_path = os.path.join(path, '../eval_data/Coffee')

    mat_file = sio.loadmat(eval_path)
    ts = mat_file['data']
    mp = mat_file['matrixProfile']
    mpi = mat_file['profileIndex'].astype(int).squeeze() - 1
    window_size = int(mat_file['subLen'][0][0])

    c = ts_cluster(ts, window_size, mp, mpi, 4)


