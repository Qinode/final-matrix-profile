
import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
from src.mds.util import pattern_mds

if __name__ == '__main__':
    data_sets = os.listdir('../eval_data/all')

    bit_dict = {'WormsTwoClass': 6, 'SonyAIBORobotSurface': 3, 'WordsSynonyms': 4, 'Car': 5, 'Wine': 7,
                'PhalangesOutlinesCorrect': 5, 'MiddlePhalanxOutlineAgeGroup': 6, 'FaceFour': 3, '50words': 4,
                'Cricket_Y': 3, 'ShapeletSim': 3, 'Adiac': 6, 'OSULeaf': 3, 'TwoLeadECG': 6, 'Ham': 4, 'SwedishLeaf': 5,
                'Strawberry': 7, 'Herring': 4, 'Trace': 7, 'DistalPhalanxOutlineAgeGroup': 5, 'Gun_Point': 6,
                'Lighting7': 3, 'CBF': 3, 'Beef': 6, 'Coffee': 6, 'Meat': 7, 'DistalPhalanxTW': 5, 'ArrowHead': 5,
                'ItalyPowerDemand': 5, 'DistalPhalanxOutlineCorrect': 5, 'OliveOil': 5, 'Cricket_Z': 3, 'Lighting2': 3,
                'MiddlePhalanxOutlineCorrect': 5, 'Cricket_X': 3, 'FISH': 6, 'ToeSegmentation1': 3, 'Worms': 6,
                'Plane': 5, 'ToeSegmentation2': 3, 'ProximalPhalanxOutlineCorrect': 6, 'SonyAIBORobotSurfaceII': 3,
                'ProximalPhalanxTW': 5, 'MiddlePhalanxTW': 5, 'BeetleFly': 3, 'BirdChicken': 4, 'ECG200': 4,
                'Earthquakes': 3, 'DiatomSizeReduction': 7, 'MedicalImages': 5, 'ECGFiveDays': 7,
                'synthetic_control': 3, 'MoteStrain': 6, 'ProximalPhalanxOutlineAgeGroup': 6}

    for d in data_sets:
        # if d not in ['Wine.mat']:
        #     continue

        file = sio.loadmat('../eval_data/all/{}'.format(d))
        window_size = file['subLen'][0][0]
        data = file['data']
        tp = file['labIdx'].squeeze() - 1

        result = sio.loadmat('../eval_result/result/{}/saved-data/dnorm-{}bits.mat'.format(d[:-4], bit_dict[d[:-4]]))
        idx_bitsave = result['idx_bitsave'].reshape(-1, 3)

        cut_off = np.where(np.diff(idx_bitsave[:, 1]) > 0)[0]
        cut_off = cut_off[0] if cut_off.shape[0] != 0 else idx_bitsave.shape[0]

        idx = idx_bitsave[:, 0][:cut_off].astype(np.int)

        bad_label = []
        for i in idx:
            if np.any(np.abs(i - tp) < 0.2 * window_size):
                bad_label.append(1)
            else:
                bad_label.append(0)

        mds2 = pattern_mds(idx, window_size, data)[0]
        hit = []
        not_hit =[]
        for i in range(mds2.shape[0]):
            if bad_label[i] == 1:
                hit.append([mds2[i, 0], mds2[i, 1]])
            else:
                not_hit.append([mds2[i, 0], mds2[i, 1]])
        hit = np.array(hit)
        not_hit = np.array(not_hit)

        f = plt.figure()
        if hit.shape[0] != 0:
            plt.scatter(hit[:, 0], hit[:, 1], s=0.5 * plt.rcParams['lines.markersize'] ** 2, label='True Positive', color='r')

        if not_hit.shape[0] != 0:
            plt.scatter(not_hit[:, 0], not_hit[:, 1], s=0.5 * plt.rcParams['lines.markersize'] ** 2, label='False Positive', color='b')

        plt.legend()
        plt.title('{} - Uniform'.format(d[:-4]))
        plt.axis('off')
        f.savefig('./mds/{}/{}-Uniform.png'.format(d[:-4], d[:-4]), bbox_inches='tight')
        plt.show()
