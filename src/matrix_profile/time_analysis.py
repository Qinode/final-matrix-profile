import time
import numpy as np
from src.matrix_profile.matrixprofile import *

if __name__ == '__main__':
    # length = [1000, 5000, 10000, 25000, 50000]
    length = [100, 500, 1000, 1500, 2000]

    window_size = 20
    ntime = []
    atime = []
    otime = []

    for l in length:
        ds = np.random.random_integers(100, size=(l, 1))
        n_time = time.time()
        naive(ds, ds, window_size)
        ntime.append(time.time() - n_time)

        a_time = time.time()
        stamp(ds, ds, window_size, True)
        atime.append(time.time() - a_time)

        o_time = time.time()
        stomp(ds, ds, window_size, True)
        otime.append(time.time() - o_time)

