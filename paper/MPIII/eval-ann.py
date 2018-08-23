import argparse

import numpy as np
import time
import datetime
import scipy.io
import os

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from src.matrix_profile.matrixprofile import stomp

from paper.MPIII.visual import subsequence_selection, sax_subsequence_selection, approximate_subsequence_selction
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

    # matrix profile will be computed in scaled and z-normalised space
    # rather than original space
    # z_data = (data - np.mean(data))/(np.std(data))
    # uz_data = (z_data - np.min(z_data))/(np.max(z_data) - np.min(z_data))

    tp = mat_file['labIdx'] - 1

    p = 0.2

    window_size = int(mat_file['subLen'][0][0])

    t_min, t_max = discretization_pre(data, window_size)

    x_axis = np.arange(3, 8)
    valid_idx_arr = []

    for bits in x_axis:
        print('[{}] {} - {} bits'.format(get_timestampe(), data_name, bits))
        # 30/07/2018
        # used for testing the the behaviour of digitising the time series before the matrix profile is computed.
        # the result is in eval_fig/z_norm_mp

        # q_data = np.digitize(uz_data, interval)
        # print('[{}] {} Computing Matrix Profile.'.format(get_timestampe(), data_name))
        # mp, mpi = stomp(q_data, q_data, window_size, True)
        # print('[{}] {} Computing Matrix Profile Finished.'.format(get_timestampe(), data_name))

        # if save:
        #     scipy.io.savemat('{}/{}-mp'.format(save_path, data_name), {'mp': mp, 'mpi': mpi})

        c, h, compress_table, idx_bitsave, picking_time, process_time = approximate_subsequence_selction(data, t_min, t_max, mp, mpi, window_size, bits)
        print('[{}] {} - {}bits DNorm Compression time {}.'.format(get_timestampe(), data_name, bits, process_time[-1] - process_time[0]))

        idx_bitsave = np.array(idx_bitsave)

        cut_off = np.argmin(idx_bitsave[:, 1])

        valid_idx = idx_bitsave[:, 0][:cut_off]

        valid_idx_arr.append(valid_idx.shape[0])

        precisions, recalls = get_f(valid_idx, tp, p, window_size)

        if save:
            scipy.io.savemat('{}/saved-data/dnorm-{}bits'.format(save_path, bits),
                             {'compressible': c, 'hypothesis': h, 'compress_table': str(compress_table),
                              'idx_bitsave': idx_bitsave, 'precisions': precisions, 'recalls': recalls,
                              'picking_time': picking_time,
                              'process_time': process_time})

        plt.plot(idx_bitsave[:, 1], label='DNorm', color='C0')
        plt.axvline(cut_off, linewidth=1, label='DNorm Cut Off', color='C0')
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
        plt.legend()

        plt.title('{} bits compression\n DNorm {}'.format(bits, f1(precisions, recalls)))
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        if save:
            plt.savefig('{}/recall-precision/{} bits.png'.format(save_path, bits))
        else:
            plt.show()
        plt.clf()

    plt.plot(x_axis, valid_idx_arr, label='DNorm')
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=int, action='store', dest='part', default=-1)
    args = parser.parse_args()

    part = args.part
    if part < 0:
        raise ValueError('parameter part must be positive')

    path = os.path.dirname(os.path.abspath(__file__))
    eval_path = os.path.join(path, 'eval_data/eval_data-{}'.format(part))
    data = os.listdir(eval_path)

    sub_dirs = ['recall-precision', 'bits', 'components', 'saved-data']

    pool = ProcessPoolExecutor(7)
    futures =[]
    parmas = []

    for d in data:
        data_name = d[:-4]

        data_path = os.path.join(eval_path, data_name)
        fig_path = os.path.join(path, 'eval_result/ann_selection08-22', data_name)
        # fig_path = os.path.join(path, 'eval_fig/z_norm_mp', data_name)

        for dir_name in sub_dirs:
            if not os.path.exists(os.path.join(fig_path, dir_name)):
                os.makedirs(os.path.join(fig_path, dir_name))

        parmas.append((data_path, data_name, fig_path))

    for p in parmas:
        futures.append(pool.submit(run, p[0], p[1], True, p[2], bounded=False))

    for f in as_completed(futures):
        print(f.result())

    pool.shutdown()