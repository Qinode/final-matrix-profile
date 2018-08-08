import scipy.io as sio
import numpy as np

from paper.MPIII.util import discretization_pre, discretization, rdl


def distribution_l1(compressible, hypothesis, bits):
    bins = np.arange(bits**2) + 1
    binc = np.histogram(compressible.astype(int).reshape(-1, ), bins)[0]
    binh = np.histogram(hypothesis.astype(int).reshape(-1, ), bins)[0]

    return np.sum(np.abs(binc-binh))


def get_hits(idxs, tp, p, window_size):
    hit = 0

    for i in range(idxs.shape[0]):
        if np.min(np.abs(tp - idxs[i])) < p * window_size:
            hit += 1

    return hit


if __name__ == '__main__':
    dataset = 'Car'
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

    hc = np.array(np.meshgrid(h, c)).T.reshape(-1, 2)

    corr = np.zeros(hc.shape)

    t_min, t_max = discretization_pre(data, window_size)

    for i in range(hc.shape[0]):
        hc_idx = hc[i]
        a_hypothesis = data[hc_idx[0]: hc_idx[0]+window_size]
        a_compressible = data[hc_idx[1]: hc_idx[1]+window_size]

        quan_hypothesis = discretization(a_hypothesis, t_min, t_max, bits)
        quan_compressible = discretization(a_compressible, t_min, t_max, bits)

        n_hypothesis = (a_hypothesis - np.mean(a_hypothesis))/np.std(a_hypothesis)
        n_compressible = (a_compressible - np.mean(a_compressible))/np.std(a_compressible)

        rdl_res = rdl(quan_compressible, quan_hypothesis, bits)
        dist_l1 = distribution_l1(quan_compressible, quan_hypothesis, bits)

        l2 = np.linalg.norm(n_compressible - n_hypothesis)

        corr[i] = np.array([rdl_res, l2]).reshape(1, 2)




