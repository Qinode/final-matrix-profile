import pickle
import _thread
import time
from src.matrix_profile.matrixprofile import stomp, stamp
import numpy as np
import scipy.io as sio
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = dir_path+'/pkls'


def aug_task(data, number, window_size):
    print("Wine-{} Start".format(number))
    t_mp_start = time.time()
    mp, mpi = stomp(data, data, window_size, True)
    t_mp_end = time.time() - t_mp_start
    print("Wine-{}, MP completed".format(number))

    pkl = {'data_name': 'Wine-{}'.format(number), 'data': data, 'window_size': window_size, 'matrix_profile': mp,
           'profile_index': mpi, 'time': t_mp_end}

    pickle.dump(pkl, open('{}/Wine-{}.pkl'.format(PKL_PATH, number), 'wb'))
    print('Wine-{} - [{}]'.format(number, t_mp_end))


if __name__ == '__main__':
    import argparse

    mat_file = sio.loadmat(dir_path+'/../eval_data/Wine')
    window_size = int(mat_file['subLen'][0][0])
    data = mat_file['data']
    data = data[~np.isnan(data)]

    eval_data = './aug_wine'
    start = 1
    end = 3

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--start', type=int, action='store', dest='start', default=-1)
    # parser.add_argument('--end', type=int, action='store', dest='end', default=-1)

    # args = parser.parse_args()

    # if args.start == -1 or args.end == -1:
    #     raise ValueError("IIlegal value for start and end")

    # for i in range(args.start, args.end):
    for i in range(1, 2):
        i_data = np.tile(data, i+2)
        print('Wine-{} - {}'.format(i+1, i_data.shape))

        aug_task(i_data, i+1, window_size)
        # _thread.start_new_thread(aug_task, (i_data, i+1, window_size))

    # while True:
    #     pass
