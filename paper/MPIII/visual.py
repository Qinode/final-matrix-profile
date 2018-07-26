
from paper.MPIII.util import *

import numpy as np
import matplotlib.pyplot as plt

def subsequence_selection(time_series, t_min, t_max, mp, mpi, window_size, nums, bits):
    max_salient = np.around(time_series.shape[0] / window_size)
    mp, mpi = mp.copy(), mpi.copy()

    C, C_idx, H, H_idx = [], [], [], []
    idx_bitsave = []
    compress_table = {}
    U = time_series.shape[0] - window_size + 1

    bit_cost = U * window_size * bits

    while True and (len(C) + len(H) <= max_salient) and U >= 0:

        candidates, candidate_idxs = pick_candidates(time_series, t_min, t_max, window_size, mp, bits, nums)

        if not candidate_idxs:
            break

        best_cand, cand_idx, cand_type, compress_by = pick_best_candidates(time_series, t_min, t_max, candidates, candidate_idxs, H, bits, mpi)

        exc_start = max(0, cand_idx - (window_size // 2))
        exc_end = min(cand_idx + (window_size // 2), mp.shape[0])
        mp[exc_start: exc_end] = np.inf

        if cand_type == 0:
            H.append(best_cand)
            H_idx.append(cand_idx)

            if idx_bitsave:
                pre_cost = idx_bitsave[-1][1]
                idx_bitsave.append([cand_idx, pre_cost, cand_type])
            else:
                idx_bitsave.append([cand_idx, bit_cost, cand_type])

            U -= 1

            if cand_idx not in compress_table:
                compress_table[cand_idx] = []
        else:
            C.append(best_cand)
            C_idx.append(cand_idx)
            U -= 1
            new_cost = bit(C, H, U, window_size, bits)
            bit_cost = min(bit_cost, new_cost)

            idx_bitsave.append([cand_idx, new_cost, cand_type])

            compress_by = H_idx[compress_by]
            if compress_by not in compress_table:
                compress_table[compress_by] = [cand_idx]
            else:
                compress_table[compress_by].append(cand_idx)

    return C_idx, H_idx, compress_table, idx_bitsave


def sax_subsequence_selection(time_series, interval, t_min, t_max, mp, mpi, window_size, nums, bits):
    max_salient = np.around(time_series.shape[0] / window_size)
    mp, mpi = mp.copy(), mpi.copy()

    C, C_idx, H, H_idx = [], [], [], []
    idx_bitsave = []
    compress_table = {}
    U = time_series.shape[0] - window_size + 1

    bit_cost = U * window_size * bits

    while True and (len(C) + len(H) <= max_salient) and U >= 0:
        # print('SAX Bits: {}.'.format(bit_cost))
        candidates, candidate_idxs = sax_pick_candidates(time_series, interval, t_min, t_max, window_size, mp, nums)

        if not candidate_idxs:
            break

        best_cand, cand_idx, cand_type, compress_by = sax_pick_best_candidates(time_series, interval, t_min, t_max, candidates, candidate_idxs, H, bits, mpi)

        exc_start = max(0, cand_idx - (window_size // 2))
        exc_end = min(cand_idx + (window_size // 2), mp.shape[0])
        mp[exc_start: exc_end] = np.inf

        if cand_type == 0:
            H.append(best_cand)
            H_idx.append(cand_idx)

            if idx_bitsave:
                pre_cost = idx_bitsave[-1][1]
                idx_bitsave.append([cand_idx, pre_cost, cand_type])
            else:
                idx_bitsave.append([cand_idx, bit_cost, cand_type])

            U -= 1

            if cand_idx not in compress_table:
                compress_table[cand_idx] = []
        else:
            C.append(best_cand)
            C_idx.append(cand_idx)
            U -= 1
            new_cost = bit(C, H, U, window_size, bits)
            idx_bitsave.append([cand_idx, new_cost, cand_type])

            bit_cost = min(bit_cost, new_cost)

            compress_by = H_idx[compress_by]
            if compress_by not in compress_table:
                compress_table[compress_by] = [cand_idx]
            else:
                compress_table[compress_by].append(cand_idx)

    return C_idx, H_idx, compress_table, idx_bitsave


if __name__ == "__main__":
    import scipy.io
    from paper.MPIII.eval import get_f, f1

    bits = range(3, 8)
    data = scipy.io.loadmat('eval_data/Plane')

    ts = data['data']
    window_size = int(data['subLen'][0][0])
    mp = data['matrixProfile']
    mpi = data['profileIndex'] - 1
    tp = data['labIdx'] - 1

    t_min, t_max = discretization_pre(ts, window_size)

    for b in [4]:
        interval = sax_discretization_pre(ts, b, bounded=False)
        c, h, compress_table, idx_bitsave = sax_subsequence_selection(ts, interval, t_min, t_max, mp, mpi, window_size, 10, b)
        idx_bitsave = np.array(idx_bitsave)

        cut_off = np.where(np.diff(idx_bitsave[:, 1]) > 0)[0]

        if cut_off.shape[0] == 0:
            cut_off = idx_bitsave[:, 1].shape[0]
        else:
            cut_off = cut_off[0]

        valid_idx = idx_bitsave[:, 0][:cut_off]
        precisions, recalls = get_f(valid_idx, tp, 0.2, window_size)

        plt.plot(recalls, precisions)
        plt.title('{} bits compression\n Gaussian {}'.format(b, f1(precisions, recalls)))
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.show()
