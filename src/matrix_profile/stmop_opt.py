import time
import numpy as np
from src.matrix_profile.matrixprofile import *

if __name__ == '__main__':
    length = [100, 500, 1000, 2000, 4000, 6000]

    window_size = 20
    sotime = []
    otime = []

    for l in length:
        ds = np.random.random_integers(100, size=(l, 1))
        # n_time = time.time()
        # naive(ds, ds, window_size)
        # ntime.append(time.time() - n_time)

        so_time = time.time()
        slow_stomp(ds, ds, window_size, True)
        sotime.append(time.time() - so_time)

        o_time = time.time()
        stomp(ds, ds, window_size, True)
        otime.append(time.time() - o_time)
