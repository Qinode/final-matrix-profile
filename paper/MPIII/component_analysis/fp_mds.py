from paper.MPIII.util import *
from random import shuffle
import ast
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from src.mds.util import pattern_mds
from paper.MPIII.eval import validation


def subsequence_process(sub, min, max):
    sub = (sub - np.mean(sub))/np.std(sub)
    sub = (sub - min)/(max - min)
    return sub


def mds_plot(dataset_name, bits, data_path, mat_path, dist):

    raw_data = sio.loadmat(data_path)
    data = raw_data['data']
    tp = raw_data['labIdx'] - 1
    p = 0.2
    window_size = int(raw_data['subLen'][0][0])
    mat_file = sio.loadmat(mat_path)
    compressible = mat_file['compressible']
    hypothesis = mat_file['hypothesis']
    idx_bitsave = mat_file['idx_bitsave']

    try:
        cut_off = np.where(np.diff(idx_bitsave[:, 1]) > 0)[0][0]
    except IndexError as error:
        print('{} - {} - {}'.format(dataset_name, bits, dist))
        return

    valid_idx = idx_bitsave[:, 0][:cut_off]
    valid_h = np.intersect1d(valid_idx, hypothesis)
    valid_c = np.intersect1d(valid_idx, compressible)

    assert(valid_h.shape[0] + valid_c.shape[0] == valid_idx.shape[0])

    Y, _ = pattern_mds(valid_idx.astype(int), window_size, data)

    fig, ax = plt.subplots()

    h, c, fp_h, fp_c = [], [], [], []
    for i in range(valid_idx.shape[0]):
        if idx_bitsave[i, 2] == 0:
            if validation(valid_idx[i], tp, p, window_size):
                h.append([Y[i, 0], Y[i, 1]])
            else:
                fp_h.append([Y[i, 0], Y[i, 1]])
        else:
            if validation(valid_idx[i], tp, p, window_size):
                c.append([Y[i, 0], Y[i, 1]])
            else:
                fp_c.append([Y[i, 0], Y[i, 1]])

    h = np.array(h)
    c = np.array(c)
    fp_h = np.array(fp_h)
    fp_c = np.array(fp_c)

    if h.shape[0] != 0:
        ax.scatter(h[:, 0], h[:, 1], marker='x', color='C0', label='hypothesis')

    if c.shape[0] != 0:
        ax.scatter(c[:, 0], c[:, 1], marker='o', color='C1', label='compressible')

    if fp_h.shape[0] != 0:
        ax.scatter(fp_h[:, 0], fp_h[:, 1], marker='x', color='C2', label='FP hypothesis')

    if fp_c.shape[0] != 0:
        ax.scatter(fp_c[:, 0], fp_c[:, 1], marker='o', color='C3', label='FP compressible')

    ax.legend(loc=2)
    ax.grid(True)
    plt.title('MDS Visual of {}'.format(dataset_name))
    plt.savefig('./result/{}/{}-{}-{}bits-mds.png'.format(dataset_name, dist, dataset_name, bits))
    plt.clf()


if __name__ == '__main__':
    best_bit = {
        '50words': [('dnorm', 4), ('gaussian', 3)],
        'Adiac': [('dnorm', 6), ('gaussian', 6)],
        'ArrowHead': [('dnorm', 5), ('gaussian', 4)],
        'Beef': [('dnorm', 6), ('gaussian', 4)],
        'BeetleFly': [('dnorm', 3), ('gaussian', 3)],
        'BirdChicken': [('dnorm', 4), ('gaussian', 3)],
        'Car': [('dnorm', 5), ('gaussian', 4)],
        'CBF': [('dnorm', 3), ('gaussian', -1)],
        'Coffee': [('dnorm', 6), ('gaussian', 6)],
        'Cricket_X': [('dnorm', 3), ('gaussian', 3)],
        'Cricket_Y': [('dnorm', 3), ('gaussian', 3)],
        'Cricket_Z': [('dnorm', 3), ('gaussian', 3)],
        'DiatomSizeReduction': [('dnorm', 7), ('gaussian', 7)],
        'DistalPhalanxOutlineAgeGroup': [('dnorm', 5), ('gaussian', 4)],
        'DistalPhalanxOutlineCorrect': [('dnorm', 5), ('gaussian', 3)],
        'DistalPhalanxTW': [('dnorm', 5), ('gaussian', 4)],
        'Earthquakes': [('dnorm', 3), ('gaussian', 5)],
        'ECG200': [('dnorm', 4), ('gaussian', 3)],
        'ECGFiveDays': [('dnorm', 7), ('gaussian', 6)],
        'FaceFour': [('dnorm', 3), ('gaussian', 3)],
        'FISH': [('dnorm', 6), ('gaussian', 7)],
        'Gun_Point': [('dnorm', 6), ('gaussian', 4)],
        'Ham': [('dnorm', 4), ('gaussian', 3)],
        'Herring': [('dnorm', 4), ('gaussian', 4)],
        'ItalyPowerDemand': [('dnorm', 5), ('gaussian', 4)],
        'Lighting2': [('dnorm', 3), ('gaussian', 3)],
        'Lighting7': [('dnorm', 3), ('gaussian', 3)],
        'Meat': [('dnorm', 7), ('gaussian', 7)],
        'MedicalImages': [('dnorm', 5), ('gaussian', 5)],
        'MiddlePhalanxOutlineAgeGroup': [('dnorm', 5), ('gaussian', 5)],
        'MiddlePhalanxOutlineCorrect': [('dnrom', 5), ('gaussian', 5)],
        'MiddlePhalanxTW': [('dnorm', 5), ('gaussian', 5)],
        'MoteStrain': [('dnorm', 6), ('gaussian', 5)],
        'OliveOil': [('dnorm', 7), ('gaussian', 7)],
        'OSULeaf': [('dnorm', 3), ('gaussian', 3)],
        'Plane': [('dnorm', 5), ('gaussian', 4)],
        'ProximalPhalanxOutlineAgeGroup': [('dnorm', 6), ('gaussian', 5)],
        'ProximalPhalanxOutlineCorrect': [('dnorm', 6), ('gaussian', 5)],
        'ProximalPhalanxTW': [('dnorm', 6), ('gaussian', 5)],
        'SonyAIBORobotSurface': [('dnorm', 3), ('gaussian', 3)],
        'SonyAIBORobotSurfaceII': [('dnorm', 3), ('gaussian', 3)],
        'Strawberry': [('dnorm', 7), ('gaussian', 7)],
        'SwedishLeaf': [('dnorm', 5), ('gaussian', 4)],
        'ToeSegmentation1': [('dnorm', 3), ('gaussian', 3)],
        'ToeSegmentation2': [('dnorm', 3), ('gaussian', 3)],
        'Trace': [('dnrom', 7), ('gaussian', 7)],
        'TwoLeadECG': [('dnorm', 7), ('gaussian', 7)],
        'Wine': [('dnorm', 7), ('gaussian', 7)],
        'WordsSynonyms': [('dnorm', 3), ('gaussian', 3)],
        'Worms': [('dnorm', 6), ('gaussian', 5)],
        'WormsTwoClass': [('dnorm', 6), ('gaussian', 5)],
        'ShapeletSim': [('dnorm', -1), ('gaussian', -1)],
        'synthetic_control': [('dnrom', -1), ('gaussian', -1)]
    }

    path = os.path.dirname(os.path.abspath(__file__))
    eval_path = os.path.join(path, '../eval_fig/normal-1-0')
    dir_name = os.listdir(eval_path)
    # shuffle(dir_name)
    os.makedirs('./result/')
    for d in dir_name[:]:
        dataset_name = os.path.basename(d)
        os.makedirs('./result/{}'.format(dataset_name))

        print(dataset_name)
        data_path = os.path.join(path, '../eval_data/{}.mat'.format(dataset_name))

        if best_bit[d][0][1] != -1:
            mat_path = os.path.join(eval_path, d, 'saved-data', 'gaussian-{}bits.mat'.format(best_bit[d][0][1]))
            mds_plot(dataset_name, best_bit[d][0][1], data_path, mat_path, best_bit[d][0][0])

        if best_bit[d][1][1] != -1:
            mat_path = os.path.join(eval_path, d, 'saved-data', 'dnorm-{}bits.mat'.format(best_bit[d][1][1]))
            mds_plot(dataset_name, best_bit[d][1][1], data_path, mat_path, best_bit[d][1][0])
