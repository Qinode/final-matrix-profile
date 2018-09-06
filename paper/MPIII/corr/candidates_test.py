import scipy.io as sio
import os
import numpy as np

from paper.MPIII.util import discretization_pre, pick_candidates
from paper.MPIII.corr.rdl_distribution import get_hits

# this script test the coverage of cs against hc
#   * cs is the first n candidates selected from matrix profile
#   * hc is the n candidates selected by best_candidate selection algorithm
#
# the script by experiment proves that the coverage of cs is very similar to hc,
# which can be extracted from the matrix profile at the very first time and used
# to build the approximate nearest neighbor search.

def f(data_name, raw_data, saved_data, bits):
    mp = raw_data['matrixProfile']
    mpi = raw_data['profileIndex'] - 1
    window_size = raw_data['subLen'][0][0]
    tp = raw_data['labIdx']

    p = 0.2

    data = raw_data['data']
    h = saved_data['hypothesis'].transpose()
    c = saved_data['compressible'].transpose()

    hc = np.r_[h, c]
    hc_hits = get_hits(hc, tp, p, window_size)
    print('{} - HC Candidates: {} TP: {}, Hit: {}'.format(data_name, hc.shape[0], tp.shape[0], hc_hits))

    t_min, t_max = discretization_pre(data, window_size)
    candidates = pick_candidates(data, t_min, t_max, window_size, mp, bits, hc.shape[0])
    cs = np.array(candidates[1])
    cs_hits = get_hits(cs, tp, p, window_size)
    print('{} - CS Candidates: {} TP: {}, Hit: {}'.format(data_name, cs.shape[0], tp.shape[0], cs_hits))
    intersect_size = np.intersect1d(cs, hc).shape
    print('Intersection Size: {}'.format(intersect_size))

    return data_name, tp.shape[0], hc.shape[0], cs.shape[0], hc_hits, cs_hits, intersect_size[0]


if __name__ == '__main__':
    datasets = os.listdir('../eval_data/all')
    header = ['data', 'tp', 'bc candidates', 'ac candidates', 'bc htis', 'ac hits', 'intersection']
    result = []
    result.append(header)
    for ds in datasets:
        dataset = ds[:-4]
        bits = 4
        saved_data = sio.loadmat('../eval_result/result/{}/saved-data/dnorm-{}bits'.format(dataset, bits))
        raw_data = sio.loadmat('../eval_data/all/{}'.format(dataset))

        result.append(f(dataset, raw_data, saved_data, bits))

