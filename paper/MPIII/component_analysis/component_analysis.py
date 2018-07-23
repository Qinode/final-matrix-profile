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


def component_analysis(dataset_name, bits, data_path, mat_path):

    os.makedirs('./result/gaussian-{}-{}bits'.format(dataset_name, bits))

    raw_data = sio.loadmat(data_path)
    data = raw_data['data']
    window_size = int(raw_data['subLen'][0][0])
    mat_file = sio.loadmat(mat_path)
    compressible = mat_file['compressible']
    hypothesis = mat_file['hypothesis']
    idx_bitsave = mat_file['idx_bitsave']
    compress_table = ast.literal_eval(mat_file['compress_table'][0])

    interval = sax_discretization_pre(data, bits)
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

    for i, h in enumerate(valid_h):
        compressibles = compress_table[h]

        hypothesis = subsequence_process(data[i: i+window_size], min, max)
        plt.plot(hypothesis, color='C1', label='Hypothesis')

        for c in compressibles:
            if c in valid_c:
                a_compressible = subsequence_process(data[c: c+window_size], min, max)
                plt.plot(a_compressible, color='C2', label='Compressible')

        for x in interval[1:]:
            plt.axhline(x, linewidth=1, color='C0')

        plt.legend()
        plt.savefig('./result/gaussian-{}-{}bits/{}-{}-component.png'.format(dataset_name, bits, i, h))
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
    eval_path = os.path.join(path, '../eval-fig/normal')
    dir_name = os.listdir(eval_path)
    shuffle(dir_name)
    for d in dir_name[:10]:
        dataset_name = os.path.basename(d)
        if dataset_name in gaussian:
            data_path = os.path.join(path, '../eval_data/{}.mat'.format(dataset_name))
            mat_path = os.path.join(eval_path, d, 'saved-data', 'gaussian-{}bits.mat'.format(gaussian[d]))
            component_analysis(dataset_name, gaussian[d], data_path, mat_path)
        else:
            pass
