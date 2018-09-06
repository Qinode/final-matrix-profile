
import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
from src.mds.util import pattern_mds

if __name__ == '__main__':
    data_sets = os.listdir('../eval_data/all')

    bit_dict = {'WormsTwoClass': 5, 'SonyAIBORobotSurface': 3, 'WordsSynonyms': 3, 'Car': 4, 'Wine': 6,
                'PhalangesOutlinesCorrect': 4, 'MiddlePhalanxOutlineAgeGroup': 5, 'FaceFour': 3, '50words': 3,
                'Cricket_Y': 3, 'ShapeletSim': 3, 'Adiac': 6, 'OSULeaf': 3, 'TwoLeadECG': 6, 'Ham': 3, 'SwedishLeaf': 4,
                'Strawberry': 7, 'Herring': 4, 'Trace': 7, 'DistalPhalanxOutlineAgeGroup': 4, 'Gun_Point': 4,
                'Lighting7': 3, 'CBF': 3, 'Beef': 4, 'Coffee': 6, 'Meat': 7, 'DistalPhalanxTW': 4, 'ArrowHead': 4,
                'ItalyPowerDemand': 4, 'DistalPhalanxOutlineCorrect': 3, 'OliveOil': 7, 'Cricket_Z': 3, 'Lighting2': 3,
                'MiddlePhalanxOutlineCorrect': 5, 'Cricket_X': 3, 'FISH': 7, 'ToeSegmentation1': 3, 'Worms': 5,
                'Plane': 4, 'ToeSegmentation2': 3, 'ProximalPhalanxOutlineCorrect': 5, 'SonyAIBORobotSurfaceII': 3,
                'ProximalPhalanxTW': 5, 'MiddlePhalanxTW': 5, 'BeetleFly': 3, 'BirdChicken': 3, 'ECG200': 3,
                'Earthquakes': 5, 'DiatomSizeReduction': 7, 'MedicalImages': 5, 'ECGFiveDays': 6,
                'synthetic_control': 3, 'MoteStrain': 5, 'ProximalPhalanxOutlineAgeGroup': 5}

    for d in data_sets:
        # if d not in ['Wine.mat', 'TwoLeadECG.mat', 'FaceFour.mat']:
        #     continue

        file = sio.loadmat('../eval_data/all/{}'.format(d))
        window_size = file['subLen'][0][0]
        data = file['data']
        tp = file['labIdx'].squeeze() - 1

        result = sio.loadmat('../eval_result/result/{}/saved-data/norm-{}bits.mat'.format(d[:-4], bit_dict[d[:-4]]))
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

        if mds2.shape[1] < 2:
            f = plt.figure()
            plt.title('{} - Gaussian'.format(d[:-4]))
            plt.axis('off')
            f.savefig('./mds/{}/{}-Gaussian.png'.format(d[:-4], d[:-4]), bbox_inches='tight')
            plt.show()
            continue

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
        plt.title('{} - Gaussian'.format(d[:-4]))
        plt.axis('off')
        f.savefig('./mds/{}/{}-Gaussian.png'.format(d[:-4], d[:-4]), bbox_inches='tight')
        plt.show()