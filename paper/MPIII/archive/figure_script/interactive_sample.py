import scipy.io
from paper.MPIII.util import *
from src.matrix_profile.matrixprofile import *
from src.mds.util import *
from paper.MPIII.visual import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    window_size = 120
    bits = 4

    data = scipy.io.loadmat('../data/data')['data']
    data = data[~np.isnan(data)]

    mp, mpi = stomp(data, data, window_size, True)

    t_min, t_max = discretization_pre(data, window_size)

    c, h, compress_table = subsequence_selection(data, t_min, t_max, mp, mpi, window_size, 10, bits)

    cands_co, _ = pattern_mds(c + h, window_size, data)

    plt.scatter(cands_co[:, 0], -cands_co[:, 1])
    plt.title('Points selected from compressing.')
    plt.show()