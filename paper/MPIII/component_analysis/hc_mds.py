from paper.MPIII.util import *
from random import shuffle
import ast
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from src.mds.util import pattern_mds


def subsequence_process(sub, min, max):
    sub = (sub - np.mean(sub))/np.std(sub)
    sub = (sub - min)/(max - min)
    return sub


def mds_plot(dataset_name, bits, data_path, mat_path):

    raw_data = sio.loadmat(data_path)
    data = raw_data['data']
    window_size = int(raw_data['subLen'][0][0])
    mat_file = sio.loadmat(mat_path)
    compressible = mat_file['compressible']
    hypothesis = mat_file['hypothesis']
    idx_bitsave = mat_file['idx_bitsave']

    try:
        cut_off = np.where(np.diff(idx_bitsave[:, 1]) > 0)[0][0]
    except IndexError as error:
        print('{} - {}'.format(dataset_name, bits))
        return

    valid_idx = idx_bitsave[:, 0][:cut_off]
    valid_h = np.intersect1d(valid_idx, hypothesis)
    valid_c = np.intersect1d(valid_idx, compressible)

    assert(valid_h.shape[0] + valid_c.shape[0] == valid_idx.shape[0])

    Y, _ = pattern_mds(valid_idx.astype(int), window_size, data)

    fig, ax = plt.subplots()

    h, c = [], []
    for i in range(valid_idx.shape[0]):
        if idx_bitsave[i, 2] == 0:
            h.append([Y[i, 0], Y[i, 1]])
        else:
            c.append([Y[i, 0], Y[i, 1]])

    h = np.array(h)
    c = np.array(c)

    ax.scatter(h[:, 0], h[:, 1], marker='x', color='C0', label='hypothesis')
    ax.scatter(c[:, 0], c[:, 1], marker='o', color='C1', label='compressible')

    ax.legend(loc=2)
    ax.grid(True)
    plt.title('MDS Visual of {}'.format(dataset_name))
    plt.savefig('./result/gaussian/{}-{}bits-mds.png'.format(dataset_name, bits))
    plt.clf()


if __name__ == '__main__':
    gaussian = {'Adiac': 6,
                'BirdChicken': 3,
                'DistalPhalanxOutlineAgeGroup': 4,
                'ECGFiveDays': 6,
                'FISH': 7,
                'Herring': 4,
                'ItalyPowerDemand': 4,
                'Lighting7': 3,
                'MedicalImages': 5,
                'MiddlePhalanxOutlineAgeGroup': 5,
                'MoteStrain': 5,
                'Plane': 4,
                'ProximalPhalanxOutlineCorrect': 5,
                'ProximalPhalanxTW': 5,
                'SwedishLeaf': 4,
                'Trace': 7,
                'TwoLeadECG': 6,
                'Worms': 5,
                'WormsTwoClass': 5
            }

    path = os.path.dirname(os.path.abspath(__file__))
    eval_path = os.path.join(path, '../eval_fig/normal')
    dir_name = os.listdir(eval_path)
    # shuffle(dir_name)
    os.makedirs('./result/gaussian')
    for d in dir_name[:]:
        dataset_name = os.path.basename(d)
        if dataset_name in gaussian:
            print(dataset_name)
            data_path = os.path.join(path, '../eval_data/{}.mat'.format(dataset_name))
            mat_path = os.path.join(eval_path, d, 'saved-data', 'gaussian-{}bits.mat'.format(gaussian[d]))
            mds_plot(dataset_name, gaussian[d], data_path, mat_path)
        else:
            pass