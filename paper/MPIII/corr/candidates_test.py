import scipy.io as sio
import os
import numpy as np

from paper.MPIII.util import discretization_pre, pick_candidates
from paper.MPIII.corr.rdl_distribution import get_hits

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

    return hc_hits, cs_hits


if __name__ == '__main__':
    datasets = os.listdir('../eval_data')
    result = []
    for ds in datasets:
        dataset = ds[:-4]
        bits = 4
        saved_data = sio.loadmat('../eval_fig/normal-1-0/{}/saved-data/dnorm-{}bits'.format(dataset, bits))
        raw_data = sio.loadmat('../eval_data/{}'.format(dataset))

        hc_hits, cs_hits = f(dataset, raw_data, saved_data, bits)

        ds_res = [dataset, hc_hits, cs_hits]
        result.append(ds_res)

