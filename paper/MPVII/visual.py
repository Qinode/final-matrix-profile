import csv
from src.matrix_profile.matrixprofile import lrstomp
import scipy.io
import os
import numpy as np

if __name__ == '__main__':
    path = os.path.abspath(os.path.dirname(__file__))
    kolhs_csv = os.path.join(path, './data/04012004-.csv')
    kolhs_mat = os.path.join(path, './data/kolhs.mat')

    # read csv file and store in numpy ndarray

    # dates, index = [], []
    #
    # with open(kolhs_csv, 'r') as input:
    #     reader = csv.reader(input)
    #     header = next(reader, None)
    #     for row in reader:
    #         dates.append(row[0])
    #         index.append(int(row[1]))
    #         print('{}, {}'.format(row[0], row[1]))
    #
    # dates, index = np.array(dates), np.array(index)
    # kolhs_google_trend = np.c_[dates, index]

    kolhs = scipy.io.loadmat(kolhs_mat)['kolhs']

    time_series = kolhs[:, 1].astype(np.float)
    windos_size = 76
    m, l, r = lrstomp(time_series, time_series, windos_size, True)










