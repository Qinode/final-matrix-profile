from paper.MPIII.util import *
from random import shuffle
import ast
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def subsequence_process(sub, min, max):
    sub = (sub - np.mean(sub))/np.std(sub)
    sub = (sub - min)/(max - min)
    return sub


def component_analysis(dataset_name, bits, data_path, dnorm_mat, gaussian_mat):

    raw_data = sio.loadmat(data_path)
    data = raw_data['data']
    mpi = raw_data['profileIndex'] - 1
    window_size = int(raw_data['subLen'][0][0])

    dnorm_mat_file = sio.loadmat(dnorm_mat)
    gaussian_mat_file = sio.loadmat(gaussian_mat)

    compressible = gaussian_mat_file['compressible']
    hypothesis = gaussian_mat_file['hypothesis']
    idx_bitsave = gaussian_mat_file['idx_bitsave']
    compress_table = ast.literal_eval(gaussian_mat_file['compress_table'][0])

    dnorm_idx_bit_save = dnorm_mat_file['idx_bitsave']

    interval = sax_discretization_pre(data, bits, 'norm')
    min, max = discretization_pre(data, window_size)

    try:
        cut_off = np.where(np.diff(idx_bitsave[:, 1]) > 0)[0][0]
    except IndexError as error:
        print('{} - {}'.format(dataset_name, bits))
        return

    valid_idx = idx_bitsave[:, 0][:cut_off]
    valid_h = np.intersect1d(valid_idx, hypothesis)
    valid_c = np.intersect1d(valid_idx, compressible)

    assert(valid_h.shape[0] + valid_c.shape[0] == valid_idx.shape[0])

    compressible_idx = 271393 # int(idx_bitsave[cut_off + 1][0])
    hypothesis_idx = int(mpi[compressible_idx])

    # for key, value in compress_table.items():
    #     if compressible_idx in value:
    #         hypothesis_idx = int(key)
    #         print(hypothesis_idx)
    #         break

    print(hypothesis_idx)
    # plt.plot(discretization(data[compressible_idx:compressible_idx+window_size], min, max, bits), label='compressible')
    # plt.plot(discretization(data[hypothesis_idx:hypothesis_idx+window_size], min, max, bits), label='hypothesis')
    plt.plot(subsequence_process(data[compressible_idx:compressible_idx+window_size], min, max), label='compressible')
    plt.plot(subsequence_process(data[hypothesis_idx:hypothesis_idx+window_size], min, max), label='hypothesis')
    plt.legend()

    for i in interval[1:]:
        plt.axhline(i, linewidth=0.5)

    plt.show()

    # print('dnrom - gaussian')
    # np.set_printoptions(threshold=np.nan)
    # idx_compare = np.c_[dnorm_idx_bit_save[:, 0], idx_bitsave[:, 0], dnorm_idx_bit_save[:, 2], idx_bitsave[:, 2]]
    # print(idx_compare.astype(np.int))



if __name__ == '__main__':
    gaussian = {'Adiac': 6,
                # 'BirdChicken': 3,
                # 'DistalPhalanxOutlineAgeGroup': 4,
                # 'ECGFiveDays': 6,
                # 'FISH': 7,
                # 'Herring': 4,
                # 'ItalyPowerDemand': 4,
                # 'Lighting7': 3,
                # 'MedicalImages': 5,
                # 'MiddlePhalanxOutlineAgeGroup': 5,
                # 'MoteStrain': 5,
                # 'Plane': 4,
                # 'ProximalPhalanxOutlineCorrect': 5,
                # 'ProximalPhalanxTW': 5,
                # 'SwedishLeaf': 4,
                # 'Trace': 7,
                # 'TwoLeadECG': 6,
                # 'Worms': 5,
                # 'WormsTwoClass': 5
                }

    path = os.path.dirname(os.path.abspath(__file__))
    eval_path = os.path.join(path, '../eval_fig/normal-1-0')
    dir_name = os.listdir(eval_path)
    # shuffle(dir_name)
    for d in dir_name[:]:
        dataset_name = os.path.basename(d)
        if dataset_name in gaussian:
            data_path = os.path.join(path, '../eval_data/{}.mat'.format(dataset_name))
            dnorm_mat = os.path.join(eval_path, d, 'saved-data', 'dnorm-{}bits.mat'.format(gaussian[d]))
            gaussian_mat = os.path.join(eval_path, d, 'saved-data', 'gaussian-{}bits.mat'.format(gaussian[d]))
            component_analysis(dataset_name, gaussian[d], data_path, dnorm_mat, gaussian_mat)
        else:
            pass
