import scipy.io
from paper.MPIII.util import *
from src.mds.util import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    windows_size = 178
    bits = 4

    data = scipy.io.loadmat('../data/106')['data']
    mp = scipy.io.loadmat('../data/106_mp')['mp'].transpose()
    mpi = scipy.io.loadmat('../data/106_mpi')['mpi'].transpose()

    t_min, t_max = discretization_pre(data, windows_size)

    cands = pick_candidates(data, t_min, t_max, windows_size, mp, bits, 2027)[1]

    cands_co, _ = pattern_mds(cands, windows_size, data)

    plt.scatter(cands_co[:, 0], cands_co[:, 1])
    plt.title('Points from mp.')
    plt.show()

