import numpy as np
import time
import datetime
import scipy.io
import os

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from src.matrix_profile.matrixprofile import stomp


from paper.MPIII.visual import subsequence_selection, sax_subsequence_selection
from paper.MPIII.util import *


def get_timestampe():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')


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


def f1(precisions, recalls):

    if len(precisions) == 0 or len(recalls) == 0:
        return 0

    precision = precisions[-1]
    recall = recalls[-1]

    return np.around(2.0 / ((1.0/precision) + (1.0/recall)), decimals=3)


def run(dataset, data_name, save, save_path, bounded=False):
    mat_file = scipy.io.loadmat(dataset)
    data = mat_file['data']
    mp = mat_file['matrixProfile']
    mpi = mat_file['profileIndex'] - 1

    tp = mat_file['labIdx'] - 1

    p = 0.2

    window_size = int(mat_file['subLen'][0][0])

    t_min, t_max = discretization_pre(data, window_size)

    x_axis = np.arange(3, 8)
    valid_idx_arr = []
    s_valid_idx_arr = []

    for bits in x_axis:
        print('[{}] {} - {} bits'.format(get_timestampe(), data_name, bits))
        interval = sax_discretization_pre(data, bits, bounded=bounded)

        print('[{}] {} Computing Matrix Profile Finished.'.format(get_timestampe(), data_name))

        if save:
            scipy.io.savemat('{}/{}-mp'.format(save_path, data_name), {'mp': mp, 'mpi': mpi})

        start = time.time()
        c, h, compress_table, idx_bitsave = subsequence_selection(data, t_min, t_max, mp, mpi, window_size, 10, bits)
        end = time.time()
        print('[{}] {} DNorm Compression time {}.'.format(get_timestampe(), data_name, end - start))

        start = time.time()
        s_c, s_h, s_compress_table, s_idx_bitsave = sax_subsequence_selection(data, interval, t_min, t_max, mp, mpi, window_size, 10, bits)
        end = time.time()
        print('[{}] {} SAX Compression time {}.'.format(get_timestampe(), data_name, end - start))

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

        if save:
            scipy.io.savemat('{}/saved-data/dnorm-{}bits'.format(save_path, bits),
                             {'compressible': c, 'hypothesis': h, 'compress_table': compress_table, 'idx_bitsave': idx_bitsave, 'precisions': precisions, 'recalls': recalls})

            scipy.io.savemat('{}/saved-data/gaussian-{}bits'.format(save_path, bits),
                             {'compressible': s_c, 'hypothesis': s_h, 'compress_table': s_compress_table, 'idx_bitsave': s_idx_bitsave, 'precisions': s_precisions, 'recalls': s_recalls})

        plt.plot(idx_bitsave[:, 1], label='DNorm', color='C0')
        plt.plot(s_idx_bitsave[:, 1], label='Gaussian Norm', color='C1')
        plt.axvline(cut_off, linewidth=1, label='DNorm Cut Off', color='C0')
        plt.axvline(s_cut_off, linewidth=1, label='Gaussian Norm Cut Off', color='C1')
        plt.legend()
        plt.title('{} bits compression'.format(bits))
        plt.ylabel('Bits')
        plt.xlabel('Number of Components')
        if save:
            plt.savefig('{}/bits/{} bits.png'.format(save_path, bits))
        else:
            plt.show()
        plt.clf()

        plt.plot(recalls, precisions, label='DNorm', color='C0')
        plt.plot(s_recalls, s_precisions, label='Gaussian Norm', color='C1')
        plt.legend()

        plt.title('{} bits compression\n DNorm {} - Gaussian Norm {}'.format(bits, f1(precisions, recalls), f1(s_precisions, s_recalls)))
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        if save:
            plt.savefig('{}/recall-precision/{} bits.png'.format(save_path, bits))
        else:
            plt.show()
        plt.clf()

    plt.plot(x_axis, valid_idx_arr, label='DNorm')
    plt.plot(x_axis, s_valid_idx_arr, label='Gaussian Norm')
    plt.legend()
    plt.title('Components vs Compression Bit')
    if save:
        plt.savefig('{}/components/components-bits.png'.format(save_path))
    plt.clf()
    # plt.show()

    return '[{}] {} finished.'.format(get_timestampe(), data_name)


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#
#     run('./eval_data/Plane', 'Plane', False, '', bounded=False)

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    path = os.path.dirname(os.path.abspath(__file__))
    eval_path = os.path.join(path, 'eval_data')
    data = os.listdir(eval_path)

    sub_dirs = ['recall-precision', 'bits', 'components', 'saved-data']

    pool = ProcessPoolExecutor(7)
    futures =[]

    parmas = []

    for d in data:
        data_name = d[:-4]
        data_path = os.path.join(eval_path, data_name)
        fig_path = os.path.join(path, 'eval_fig/', data_name)

        for dir_name in sub_dirs:
            os.makedirs(os.path.join(fig_path, dir_name))

        parmas.append((data_path, data_name, fig_path))

    for p in parmas:
        futures.append(pool.submit(run, p[0], p[1], True, p[2], bounded=False))

    for f in as_completed(futures):
        print(f.result())

    pool.shutdown()