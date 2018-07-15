import scipy.io
import matplotlib.pyplot as plt
from src.matrix_profile.matrixprofile import *
from paper.MPIII.visual import *


if __name__ == '__main__':

    arrow_head = scipy.io.loadmat('ArrowHead')
    data = arrow_head['data']
    window_size = int(arrow_head['subLen'][0][0])

    py_mp, py_mpi = stomp(data, data, window_size, True)

    mp = arrow_head['matrixProfile']
    mpi = arrow_head['profileIndex'] - 1
    tp = arrow_head['labIdx'] - 1
