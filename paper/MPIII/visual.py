import pickle
import numpy as np
from src.matrix_profile.matrixprofile import stomp
from paper.MPIII.util import *


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


# kolhs data
if __name__ == "__main__":
    import pickle
    import scipy.io
    from src.matrix_profile.matrixprofile import stomp
    window_size = 78

    bits = 4
    data = scipy.io.loadmat('../MPVII/data/kolhs.mat')['kolhs']
    data = data[:, 1].astype(np.float)
    data = data[~np.isnan(data)]

    t_min, t_max = discretization_pre(data, window_size)
    mp, mpi = stomp(data, data, window_size, True)

    # scipy.io.savemat('data/106_mp.mat', {'mp': mp})
    # scipy.io.savemat('data/106_mpi.mat', {'mpi': mpi})

    c, h = subsequence_selection(data, t_min, t_max, mp.copy(), mpi.copy(), window_size, 10, bits)


# if __name__ == "__main__":
#     import pickle
#     import scipy.io
#     from src.matrix_profile.matrixprofile import stomp
#     window_size = 120
#
#     bits = 4
#     data = scipy.io.loadmat('data/data.mat')['data']
#     data = data[~np.isnan(data)]
#
#     t_min, t_max = discretization_pre(data, window_size)
#     mp, mpi = stomp(data, data, window_size, True)
#
    # scipy.io.savemat('data/106_mp.mat', {'mp': mp})
    # scipy.io.savemat('data/106_mpi.mat', {'mpi': mpi})

#    c, h = subsequence_selection(data, t_min, t_max, mp, mpi, window_size, 10, bits)
