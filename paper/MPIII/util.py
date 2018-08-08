import numpy as np
import scipy.stats


def discretization_pre(time_series, windows_size):
    t_min, t_max = np.inf, -np.inf
    for i in range(time_series.shape[0] - windows_size + 1):
        sub = time_series[i: i+windows_size]
        mean, std = np.mean(sub), np.std(sub)
        sub = (sub - mean)/std if std != 0 else sub - mean

        t_min = np.min([np.min(sub), t_min])
        t_max = np.max([np.max(sub), t_max])
    return t_min, t_max


def compress_test(data, window_size, bits, min, max, interval, compressible, hypothesis):
    compressible = data[compressible: compressible+window_size]
    hypothesis = data[hypothesis: hypothesis+window_size]

    d_compressible = sax_discretization(compressible, min, max, interval)
    d_hypothesis = sax_discretization(hypothesis, min, max, interval)

    reduced_dl = rdl(d_compressible, d_hypothesis, bits)
    return window_size * bits - reduced_dl, d_compressible, d_hypothesis


def sax_discretization_pre(time_series, bits, dist, bounded=False):
    z_time_series = (time_series - np.mean(time_series)) / np.std(time_series)
    min, max = np.min(z_time_series), np.max(z_time_series)
    unit_z_time_series = (z_time_series - min) / (max - min)
    mean, std = np.mean(unit_z_time_series), np.std(unit_z_time_series)

    if dist == 'norm':
        interval = scipy.stats.norm.ppf(np.arange(2 ** bits)/(2 ** bits), mean, std)

        if bounded:
            upper = mean + 3 * std
            lower = mean - 3 * std
            interval = (interval - lower)/(upper - lower)
    elif dist == 'beta':
        a, b, _, _ = scipy.stats.beta.fit(unit_z_time_series, loc=mean, scale=std)
        interval = scipy.stats.beta.ppf(np.arange(2 ** bits)/(2 ** bits), a, b, mean, std)
    else:
        raise ValueError("{} dist not support, only [norm, beta] are support".format(dist))

    return interval


def sax_discretization(time_series, min, max, interval):
    mean = np.mean(time_series)
    std = np.std(time_series)

    time_series = (time_series - mean)/std if std != 0 else time_series - mean
    time_series = (time_series - min)/(max - min)
    return np.digitize(time_series, interval)


def discretization(time_series, min, max, bits):
    mean = np.mean(time_series)
    std = np.std(time_series)

    if std != 0:
        time_series = (time_series - mean)/std
    else:
        time_series = time_series - mean

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

    threshold_factor = 0.0

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

        if bit_save_hypo > (best_com * (1 + threshold_factor)):
            candidates_table[i][0] = bit_save_hypo
            candidates_table[i][1] = 0
        else:
            candidates_table[i][0] = best_com
            candidates_table[i][1] = 1

    best_candidate = np.argmax(candidates_table[:, 0])

    return candidates[best_candidate], candidate_idx[best_candidate], candidates_table[best_candidate][1], compressed_by


def sax_pick_candidates(time_series, interval, t_min, t_max, window_size, mp, nums):
    mp = mp.copy()
    candidate_idxs = []
    candidates = []

    for i in range(nums):
        candidate_idx = np.argmin(mp)
        if mp[candidate_idx] == np.inf:
            break
        candidate = time_series[candidate_idx: candidate_idx + window_size]
        candidates.append(sax_discretization(candidate, t_min, t_max, interval))
        candidate_idxs.append(candidate_idx)

        exc_start = max(0, candidate_idx - (window_size // 2))
        exc_end = min(candidate_idx + (window_size // 2), mp.shape[0])

        mp[exc_start: exc_end] = np.inf

    return candidates, candidate_idxs


# 0 for hypothesis, 1 for compressible
def sax_pick_best_candidates(time_sereis, interval, t_min, t_max, candidates, candidate_idx, hypothesis, bits, mpi):
    candidates_table = np.full(((len(candidate_idx), 2)), -np.inf)

    threshold_factor = 0.0

    for i in range(len(candidates)):
        nn_idx = int(mpi[candidate_idx[i]])
        nn = time_sereis[nn_idx:nn_idx + candidates[i].shape[0]]
        nn = sax_discretization(nn, t_min, t_max, interval)
        bit_save_hypo = nn.shape[0] * bits - rdl(nn, candidates[i], bits)

        best_com = np.inf
        compressed_by = 0
        for h_idx, h in enumerate(hypothesis):
            com_bit = rdl(candidates[i], h, bits)
            if com_bit < best_com:
                best_com = com_bit
                compressed_by = h_idx

        best_com = candidates[i].shape[0] * bits - best_com

        if bit_save_hypo > (best_com * (1 + threshold_factor)):
            candidates_table[i][0] = bit_save_hypo
            candidates_table[i][1] = 0
        else:
            candidates_table[i][0] = best_com * (1 + threshold_factor)
            candidates_table[i][1] = 1

    best_candidate = np.argmax(candidates_table[:, 0])

    return candidates[best_candidate], candidate_idx[best_candidate], candidates_table[best_candidate][1], compressed_by

