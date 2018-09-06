# this script plots example of good and bad multi-dimension scaling scatter

import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
from src.mds.util import pattern_mds

if __name__ == '__main__':
    data_sets = os.listdir('../eval_data/all')

    for d in data_sets:
        if not os.path.exists('./mds/{}'.format(d[:-4])):
            os.makedirs('./mds/{}'.format(d[:-4]))

        # if d not in ['Wine.mat', 'TwoLeadECG.mat', 'FaceFour.mat']:
        #     continue

        file = sio.loadmat('../eval_data/all/{}'.format(d))
        window_size = file['subLen'][0][0]
        data = file['data']
        tp = file['labIdx'].squeeze() - 1

        mds = pattern_mds(tp, window_size, data)[0]

        f = plt.figure()
        plt.scatter(mds[:, 0], mds[:, 1], s=0.5 * plt.rcParams['lines.markersize'] ** 2)
        plt.title(d[:-4])
        plt.axis('off')
        f.savefig('./mds/{}/{}.png'.format(d[:-4], d[:-4]), bbox_inches='tight')
        plt.show()

        randidx = np.random.permutation(data.shape[0] - window_size + 1)
        randidx = randidx[:tp.shape[0]]

        bad_label = []
        for i in randidx:
            if np.any(np.abs(i - tp) < 0.2 * window_size):
                bad_label.append(1)
            else:
                bad_label.append(0)

        mds2 = pattern_mds(randidx, window_size, data)[0]
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
        plt.scatter(hit[:, 0], hit[:, 1], s=0.5 * plt.rcParams['lines.markersize'] ** 2, label='True Positive', color='r')
        plt.scatter(not_hit[:, 0], not_hit[:, 1], s=0.5 * plt.rcParams['lines.markersize'] ** 2, label='False Positive', color='b')
        plt.legend()
        plt.title('{} - Random'.format(d[:-4]))
        plt.axis('off')
        f.savefig('./mds/{}/{}-Random.png'.format(d[:-4], d[:-4]), bbox_inches='tight')
        plt.show()
