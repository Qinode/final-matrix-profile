import scipy.io
import matplotlib.pyplot as plt
from paper.MPIII.visual import *


def get_f(idxs, tp, p, window_size):
    precision = []
    recall = []
    hit = 0

    for i in range(idxs.shape[0]):
        if np.min(np.abs(tp - idxs[i])) < p * window_size:
            hit += 1

        precision.append(hit / (i + 1))
        recall.append(hit / tp.shape[0])

    return np.array(precision), np.array(recall)


if __name__ == '__main__':

    arrow_head = scipy.io.loadmat('ArrowHead')
    data = arrow_head['data']

    mp = arrow_head['matrixProfile']
    mpi = arrow_head['profileIndex'] - 1
    tp = arrow_head['labIdx'] - 1

    p = 0.2

    window_size = int(arrow_head['subLen'][0][0])
    t_min, t_max = discretization_pre(data, window_size)

    x_axis = np.arange(3, 9)
    valid_idx_arr = []
    s_valid_idx_arr = []

    bits = 6
    print('{} bits'.format(bits))
    interval = sax_discretization_pre(data, bits)

    c, h, compress_table, idx_bitsave = subsequence_selection(data, t_min, t_max, mp, mpi, window_size, 10, bits)
    s_c, s_h, s_compress_table, s_idx_bitsave = sax_subsequence_selection(data, interval, t_min, t_max, mp, mpi, window_size, 10, bits)

    idx_bitsave = np.array(idx_bitsave)
    s_idx_bitsave = np.array(s_idx_bitsave)
    cut_off = np.where(np.diff(idx_bitsave[:, 1]) > 0)[0]

    if cut_off.shape[0] == 0:
        cut_off = idx_bitsave[:, 1].shape[0]
    else:
        cut_off = cut_off[0]

    s_cut_off = np.where(np.diff(s_idx_bitsave[:, 1]) > 0)[0]

    if s_cut_off.shape[0] == 0:
        s_cut_off = s_idx_bitsave[:, 1].shape[0]
    else:
        s_cut_off = s_cut_off[0]

    valid_idx = idx_bitsave[:, 0][:cut_off]
    s_valid_idx = s_idx_bitsave[:, 0][:s_cut_off]

    valid_idx_arr.append(valid_idx.shape[0])
    s_valid_idx_arr.append(s_valid_idx.shape[0])

    precisions, recalls = get_f(valid_idx, tp, p, window_size)
    s_precisions, s_recalls = get_f(s_valid_idx, tp, p, window_size)

    plt.plot(idx_bitsave[:, 1], label='DNorm')
    plt.plot(s_idx_bitsave[:, 1], label='Gaussian Norm')
    plt.axvline(cut_off, linewidth=1, label='DNorm Cut Off', color='C0')
    plt.axvline(s_cut_off, linewidth=1, label='Gaussian Norm Cut Off', color='C1')
    plt.legend()
    plt.title('{} bits compression'.format(bits))
    plt.ylabel('Bits')
    plt.xlabel('Number of Components')
    # plt.savefig('./figure/bits/{} bits.png'.format(bits))
    plt.show()

    plt.plot(s_recalls, s_precisions, label='Gaussian Norm')
    plt.plot(recalls, precisions, label='DNorm')
    plt.legend()
    plt.title('{} bits compression'.format(bits))
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    # plt.savefig('./figure/recall-precision/{} bits.png'.format(bits))
    plt.show()

