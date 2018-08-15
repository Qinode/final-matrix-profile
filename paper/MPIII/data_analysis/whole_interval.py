import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

from paper.MPIII.util import sax_discretization_pre, discretization_pre

if __name__ == '__main__':
    datasets = os.listdir('../eval_data')

    targets = []
    targets += ['Cricket_X', 'Cricket_Y', 'Cricket_Z']

    for dataset in datasets:
        dataset = dataset[:-4]

        if len(targets) != 0 and dataset not in targets:
            continue

        bits = 4

        mat_file = sio.loadmat('../eval_data/{}'.format(dataset))
        saved_data = sio.loadmat('../eval_fig/normal-1-0/{}/saved-data/gaussian-{}bits'.format(dataset, bits))

        raw_data = mat_file['data']
        window_size = mat_file['subLen'][0][0]

        interval = sax_discretization_pre(raw_data, bits, 'norm')
        raw_data = (raw_data - np.mean(raw_data))/(np.std(raw_data))
        raw_data = (raw_data - np.min(raw_data))/(np.max(raw_data) - np.min(raw_data))

        for i in interval[1:]:
            plt.axhline(i, linewidth=0.5)

        x = np.arange(raw_data.shape[0])
        plt.title(dataset)
        plt.plot(x, raw_data)
        plt.show()



