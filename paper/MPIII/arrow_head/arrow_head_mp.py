import scipy.io
from src.mds.util import *
from paper.MPIII.visual import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    arrow_head = scipy.io.loadmat('ArrowHead')
    data = arrow_head['data']

    mp = arrow_head['matrixProfile']
    mpi = arrow_head['profileIndex'] - 1
    tp = arrow_head['labIdx'] - 1

    p = 0.2

    window_size = arrow_head['subLen'][0][0]
    bits = 5

    t_min, t_max = discretization_pre(data, window_size)

    precision = []
    recall = []

    subs = np.around(data.shape[0] / window_size)

    for i in range(int(subs)):
        print('{}/{}'.format(i, subs))
        cand_ = pick_candidates(data, t_min, t_max, window_size, mp, bits, i)[1]
        hit_miss = np.zeros((i, 1))

        for j in range(i):
            if np.min(np.abs(tp - cand_[j])) < p * window_size:
                hit_miss[j] = 1

        precision.append(np.sum(hit_miss) / hit_miss.shape[0])
        recall.append(np.sum(hit_miss) / tp.shape[0])
