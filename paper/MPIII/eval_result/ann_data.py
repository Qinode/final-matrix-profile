import os
import scipy.io as sio
import numpy as np

if __name__ == '__main__':
    base = 'ann_selection08-22/{}/saved-data/dnorm-{}bits'

    data_sets = os.listdir('ann_selection08-22')
    bits = [3, 4, 5, 6, 7]

    table = {}

    for data in data_sets:
        temp_table = {}

        precisions = []
        recalls = []
        picking_times = []
        patterns = []
        times = []

        for b in bits:
            file = sio.loadmat(base.format(data, b))
            p = file['precisions'].squeeze()
            precision = p[-1] if len(p) != 0 else 0
            r = file['recalls'].squeeze()
            recall = r[-1] if len(r) != 0 else 0
            picking_time = np.sum(file['picking_time'][0])
            t = file['process_time'][0]
            time = (t[-1] - t[0]) if len(t) != 0 else 0

            idx_bitsave = file['idx_bitsave'].squeeze()
            num_of_pattern = np.argmin(idx_bitsave[:, 1])

            precisions.append(precision)
            recalls.append(recall)
            picking_times.append(picking_time)
            times.append(time)
            patterns.append(num_of_pattern)

        temp_table['precisions'] = np.array(precisions)
        temp_table['recalls'] = np.array(recalls)
        temp_table['picking_times'] = np.array(picking_times)
        temp_table['times'] = np.array(times)
        temp_table['patterns'] = np.array(patterns)
        table[data] = temp_table
