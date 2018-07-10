import numpy as np


def discretization_pre(time_series, windows_size):
    min, max = np.inf, -np.inf
    for i in range(time_series.shape[0] - windows_size + 1):
        sub = time_series[i: i+windows_size]
        mean, std = np.mean(sub), np.std(sub)
        sub = (sub - mean)/std if std != 0 else sub - mean

        min = np.min([np.min(sub), min])
        max = np.max([np.max(sub), max])
    return min, max


def discretization(time_series, min, max, bits):
    mean = np.mean(time_series)
    std = np.std(time_series)

    time_series = (time_series - mean)/std if std != 0 else time_series - mean
    return np.around(((time_series - min)/(max - min)) * (2 ** bits - 1)) + 1


def rdl(compressible, hypothesis, bits):
    assert compressible.shape == hypothesis.shape

    m = compressible.shape[0]
    diffs = m - np.count_nonzero(compressible == hypothesis)
    return diffs * (np.log2(m) + bits)


def bit(C, H, U, window_size, bits):
    hypo_size = len(H) * window_size * bits
    unexplore_size = U * window_size * bits
    reduced_size = 0

    for c in C:
        min_rdl = np.inf
        for h in H:
            min_rdl = min(min_rdl, rdl(c, h, bits))
        reduced_size += min_rdl

    return reduced_size + len(C) * np.log2(len(H)) + hypo_size + unexplore_size


def pick_candidates(time_series, t_min, t_max, window_size, mp, bits, nums):
    mp = mp.copy()
    candidate_idxs = []
    candidates = []

    for i in range(nums):
        candidate_idx = np.argmin(mp)
        if mp[candidate_idx] == np.inf:
            break
        candidate = time_series[candidate_idx: candidate_idx + window_size]
        candidates.append(discretization(candidate, t_min, t_max, bits))
        candidate_idxs.append(candidate_idx)

        exc_start = max(0, candidate_idx - (window_size // 2))
        exc_end = min(candidate_idx + (window_size // 2), mp.shape[0])

        mp[exc_start: exc_end] = np.inf

    return candidates, candidate_idxs


# 0 for hypothesis, 1 for compressible
def pick_best_candidates(time_sereis, t_min, t_max, candidates, candidate_idx, hypothesis, bits, mpi):
    candidates_table = np.full(((len(candidate_idx), 2)), -np.inf)

    for i in range(len(candidates)):
        nn_idx = int(mpi[candidate_idx[i]])
        nn = time_sereis[nn_idx:nn_idx + candidates[i].shape[0]]
        nn = discretization(nn, t_min, t_max, bits)
        bit_save_hypo = nn.shape[0] * bits - rdl(nn, candidates[i], bits)

        best_com = np.inf
        compressed_by = 0
        for h_idx, h in enumerate(hypothesis):
            com_bit = rdl(candidates[i], h, bits)
            if com_bit < best_com:
                best_com = com_bit
                compressed_by = h_idx

        best_com = candidates[i].shape[0] * bits - best_com

        if bit_save_hypo > best_com:
            candidates_table[i][0] = bit_save_hypo
            candidates_table[i][1] = 0
        else:
            candidates_table[i][0] = best_com
            candidates_table[i][1] = 1

    best_candidate = np.argmax(candidates_table[:, 0])

    return candidates[best_candidate], candidate_idx[best_candidate], candidates_table[best_candidate][1], compressed_by

