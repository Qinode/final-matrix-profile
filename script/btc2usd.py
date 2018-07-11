import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.stats import norm
import scipy.stats


from matplotlib import gridspec, mlab

from paper.MPIII.util import *
from src.matrix_profile.matrixprofile import stomp

script_path = os.path.abspath(os.path.dirname(__file__))
data = os.path.join(script_path, './data/BTCUSDT_5m/BTCUSDT_5m.csv')
mp_path = os.path.join(script_path, './data/BTCUSDT_5m/mp.pickle')
mpi_path = os.path.join(script_path, './data/BTCUSDT_5m/mpi.pickle')
price_time = os.path.join(script_path, './data/BTCUSDT_5m/data.pickle')


if __name__ == '__main__':
    # time, price = [], []
    # with open(data, newline='') as data:
    #     reader = csv.reader(data, delimiter=',')
    #     next(reader, None)
    #     for row in reader:
    #         time.append(row[0])
    #         price.append(float(row[1]))
    #
    # price = np.array(price)
    # time = np.array(time)

    window_size = 5 * 12 * 24
    data = pickle.load(open(price_time, "rb"))  # first column timestamp, second column price
    mp = pickle.load(open(mp_path, "rb"))
    mpi = pickle.load(open(mpi_path, "rb"))

    price = data[:, 1].astype(np.float)
    t_min, t_max = discretization_pre(price, window_size)

    s_mp = np.argsort(mp)
    sub1 = data[s_mp[0]: s_mp[0] + window_size][:, 1].astype(np.float)
    nn_idx = int(mpi[s_mp[0]])
    sub2 = data[nn_idx: nn_idx + window_size][:, 1].astype(np.float)
    
    sub1_4bits = discretization(sub1, t_min, t_max, bits=4)
    sub2_4bits = discretization(sub2, t_min, t_max, bits=4)

    sub1 = (sub1 - np.mean(sub1)) / np.std(sub1)
    sub2 = (sub2 - np.mean(sub2)) / np.std(sub2)
    
    scaled_sub1 = (sub1 - t_min)/(t_max - t_min)
    scaled_sub2 = (sub2 - t_min)/(t_max - t_min)

    gs = gridspec.GridSpec(1, 2,
                           width_ratios=[3, 1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # ax1.set_ylim(0, 1)
    ax1.plot(sub1)
    ax1.plot(sub2)

    mu = 0
    variance = 1
    sigma = np.sqrt(variance)
    sample = np.random.normal(loc=0, scale=1, size=10000)
    # ax2.set_ylim(-2, 2)
    ax2.hist(sample, bins=30, orientation='horizontal')

    # plt.plot(scaled_sub2)

    # plt.plot(sub1_4bits)
    # plt.plot(sub2_4bits)
    # plt.title('4bits quantized subsequences (BTC2USDT)')
    plt.show()

    scipy.stats.probplot(sub1, plot=plt)
    scipy.stats.probplot(sub2, plot=plt)
    plt.show()






