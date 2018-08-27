import os
import scipy.io as sio
import numpy as np

if __name__ == '__main__':
    base = 'result/{}/saved-data/dnorm-{}bits'

    data_sets = os.listdir('result')
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
            p = file['precisions'].reshape(-1, )
            precision = p[-1] if len(p) != 0 else 0
            r = file['recalls'].reshape(-1, )
            recall = r[-1] if len(r) != 0 else 0

            idx_bitsave = file['idx_bitsave'].reshape(-1, 3)
            cut_offs = np.where(np.diff(idx_bitsave[:, 1]) > 0)[0]
            cut_off = cut_offs[0] if cut_offs.shape[0] != 0 else idx_bitsave.shape[0]

            picking_time = np.sum(file['picking_time'][:cut_off])
            t = file['process_time'][0]
            time = (t[cut_off] - t[0]) if len(t) != 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            picking_times.append(picking_time)
            times.append(time)
            patterns.append(cut_off)

        temp_table['precisions'] = np.array(precisions)
        temp_table['recalls'] = np.array(recalls)
        temp_table['picking_times'] = np.array(picking_times)
        temp_table['times'] = np.array(times)
        temp_table['patterns'] = np.array(patterns)
        table[data] = temp_table
