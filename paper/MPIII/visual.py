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
        print("{}, {}, {}".format(len(C), len(H), U))
        print(bit_cost)
        candidates, candidate_idxs = pick_candidates(time_series, t_min, t_max, window_size, mp, bits, nums)
        if candidate_idxs == []:
            break
        best_cand, cand_idx, cand_type, compress_by = pick_best_candidates(time_series, t_min, t_max, candidates, candidate_idxs, H, bits, mpi)

        exc_start = max(0, cand_idx - (window_size // 2))
        exc_end = min(cand_idx + (window_size // 2), mp.shape[0])
        mp[exc_start: exc_end] = np.inf

        if cand_type == 0:
            H.append(best_cand)
            H_idx.append(cand_idx)
            idx_bitsave.append([cand_idx, bit_cost])
            U -= 1
        else:
            C.append(best_cand)
            C_idx.append(cand_idx)
            U -= 1
            new_cost = bit(C, H, U, window_size, bits)
            idx_bitsave.append([cand_idx, new_cost])
            if new_cost > bit_cost:
                C, C_idx = C[:-1], C_idx[: -1]
                print("Compressing Stop, bits needed {}.".format(new_cost))
                # break
            else:
                bit_cost = new_cost
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