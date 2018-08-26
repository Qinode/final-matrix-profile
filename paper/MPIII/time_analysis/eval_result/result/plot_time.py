import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    base = './Wine-{}/saved-data/dnorm{}-{}bits.mat'

    picking_times = []
    times = []
    for i in range(1, 21):
        file = sio.loadmat(base.format(i, '', 7))
        picking_times.append(np.sum(file['picking_time']))

        process_time = file['process_time'][0]
        times.append(process_time[-1] - process_time[0])

    ann_picking_times = []
    ann_times = []
    for i in range(1, 21):
        file = sio.loadmat(base.format(i, '(ann)', 7))
        ann_picking_times.append(np.sum(file['picking_time']))

        process_time = file['process_time'][0]
        ann_times.append(process_time[-1] - process_time[0])
