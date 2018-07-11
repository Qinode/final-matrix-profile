from paper.MPIII.visual import *

if __name__ == '__main__':

    arrow_head = scipy.io.loadmat('ArrowHead')
    data = arrow_head['data']

    mp = arrow_head['matrixProfile']
    mpi = arrow_head['profileIndex'] - 1
    tp = arrow_head['labIdx'] - 1

    p = 0.2

    window_size = arrow_head['subLen'][0][0]
    bits = 5

    t_min, t_max = discretization_pre(data, window_size)

    c, h, compress_table, idx_bitsave = subsequence_selection(data, t_min, t_max, mp, mpi, window_size, 10, bits)

    idx_bitsave = np.array(idx_bitsave)
    cut_off = np.where(np.diff(idx_bitsave[:, 1]) > 0)[0][0]

    valid_idx = idx_bitsave[:, 0][:cut_off]
    hit_miss = np.zeros((cut_off, 1))

    for i in range(cut_off):
        if np.min(np.abs(tp - valid_idx[i])) < p * window_size:
            hit_miss[i] = 1

    precision = np.sum(hit_miss) / hit_miss.shape[0]
    recall = np.sum(hit_miss) / tp.shape[0]

