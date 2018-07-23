import scipy.io
from src.mds.util import *
from paper.MPIII.visual import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    window_size = 178
    bits = 8

    data = scipy.io.loadmat('../data/106')['data']

    coord_idx = scipy.io.loadmat('../data/106idx')['coordIdx'] - 1
    mp = scipy.io.loadmat('../data/106_mp')['matrixProfile']
    mpi = scipy.io.loadmat('../data/106_mpi')['profileIndex']

    t_min, t_max = discretization_pre(data, window_size)

    c, h, compress_table = subsequence_selection(data, t_min, t_max, mp, mpi, window_size, 10, bits)

    com_coord = np.array(c + h)
    precision = np.setdiff1d(com_coord, coord_idx)

    # cands_co, _ = pattern_mds(c + h, window_size, data)
    #
    # plt.scatter(cands_co[:, 0], -cands_co[:, 1])
    # plt.title('Points selected from compressing.')
    # plt.show()