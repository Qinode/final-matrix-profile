from annoy import AnnoyIndex
import numpy as np

import matplotlib as plt
import scipy.io as sio
from paper.MPIII.util import discretization, discretization_pre, pick_candidates, rdl

def exact_1nn(t, n, query, query_idx):
    min_dist = np.inf
    min_idx = -1
    for i in range(n):
        item = np.array(t.get_item_vector(i))
        dist = np.linalg.norm(query - item)
        if dist < min_dist and i != query_idx:
            min_dist = dist
            min_idx = i

    return min_idx, min_dist

if __name__ == '__main__':
    dataset = 'ProximalPhalanxOutlineCorrect'
    bits = 4
    saved_data = sio.loadmat('../eval_fig/normal-1-0/{}/saved-data/dnorm-{}bits'.format(dataset, bits))
    raw_data = sio.loadmat('../eval_data/{}'.format(dataset))

    mp = raw_data['matrixProfile']
    mpi = raw_data['profileIndex'] - 1
    window_size = raw_data['subLen'][0][0]
    tp = raw_data['labIdx']

    data = raw_data['data']
    h = saved_data['hypothesis'].transpose()
    c = saved_data['compressible'].transpose()

    patterns = data.shape[0] // window_size

    t_min, t_max = discretization_pre(data, window_size)
    candidates_matrix, candidates = pick_candidates(data, t_min, t_max, window_size, mp, bits, patterns)
    ann_indexer = AnnoyIndex(window_size, metric='euclidean')

    for i in range(patterns):
        item = data[i: i+window_size]
        item = (item - np.mean(item))/np.std(item)

        ann_indexer.add_item(i, item)

    ann_indexer.build(10)

    hs, cs = 0, 0
    ann_enn = np.zeros((patterns, 2))

    for i in range(patterns):
        mpi_nn = mpi[candidates[i]][0]

        c = candidates_matrix[i]
        quan_mpi_nn = discretization(data[mpi_nn: mpi_nn+window_size], t_min, t_max, bits)
        hypo_rdl = rdl(quan_mpi_nn, c, bits)

        enn = exact_1nn(ann_indexer, patterns, ann_indexer.get_item_vector(i), i)[0]

        ann_nn = ann_indexer.get_nns_by_item(i, 2)
        if len(ann_nn) == 0:
            print('Approximate nn not found')
            hypo_bitsave = window_size * bits - hypo_rdl
            hs += 1
            print('{} is {}, saving {}'.format(i, 'hypothesis', hypo_bitsave))
            continue
        else:
            ann_nn = ann_nn[1]
            print('{}, ANN Pair {}, ENN Pair {}'.format(i, ann_nn, enn))

        ann_enn[i] = np.array([ann_nn, enn]).reshape(1, 2)

        com_rdl = rdl(candidates_matrix[ann_nn], candidates_matrix[i], bits)

        hypo_bitsave = window_size * bits - hypo_rdl
        com_bitsave = window_size * bits - com_rdl

        if hypo_bitsave > com_bitsave:
            hs += 1
            print('{} is {}, saving {}'.format(i, 'hypothesis', hypo_bitsave))
        else:
            cs += 1
            print('{} is {}, saving {}'.format(i, 'compressible', com_bitsave))

    diff = np.count_nonzero(ann_enn[:, 0] - ann_enn[:, 1])
    print('Different: {}'.format(float(diff/patterns)))





