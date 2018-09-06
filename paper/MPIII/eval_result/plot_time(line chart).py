import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    f1 = pd.read_csv('./csv/09-01/f1s09-01.csv')
    time = pd.read_csv('./csv/09-01/times09-01.csv')

    mp_patterns = f1['mp best bit patterns'][1:].astype(int).tolist()
    gmp_patterns = f1['gmp best bit patterns'][1:].astype(int).tolist()
    ann_patterns = f1['ann best bit patterns'][1:].astype(int).tolist()

    mp_picking = time['mp best bit picking time'][1:].astype(float).tolist()
    gmp_picking = time['gmp best bit picking time'][1:].astype(float).tolist()
    ann_picking = time['ann best bit picking time'][1:].astype(float).tolist()

    mp_process = time['mp best bit time'][1:].astype(float).tolist()
    gmp_process = time['gmp best bit time'][1:].astype(float).tolist()
    ann_process = time['ann best bit time'][1:].astype(float).tolist()

    mp_timing = np.c_[mp_patterns, mp_picking, mp_process]
    mp_timing = np.sort(mp_timing, axis=0)

    gmp_timing = np.c_[gmp_patterns, gmp_picking, gmp_process]
    gmp_timing = np.sort(gmp_timing, axis=0)

    ann_timing = np.c_[ann_patterns, ann_picking, ann_process]
    ann_timing = np.sort(ann_timing, axis=0)

    color = ['c', 'm', 'r']
    marker = ['s', '.', '*']

    plt.plot(mp_timing[:, 0], mp_timing[:, 1], label='Uniform', color=color[0], marker=marker[0])
    plt.plot(gmp_timing[:, 0], gmp_timing[:, 1], label='Gaussian', color=color[1], marker=marker[1])
    # plt.plot(ann_timing[:, 0], ann_timing[:, 1], label='ANN', color=color[2], marker=marker[2])
    # plt.xscale('log')
    plt.xlabel('Number of patterns')
    plt.ylabel('Time (Second)')
    plt.legend()
    plt.show()

    # plt.plot(mp_timing[:, 0], mp_timing[:, 2], label='Uniform')
    # plt.plot(gmp_timing[:, 0], gmp_timing[:, 2], label='Gaussian')
    # plt.plot(ann_timing[:, 0], ann_timing[:, 2], label='ANN')
    # plt.xlabel('Number of patterns')
    # plt.ylabel('Time (Second)')
    # plt.legend()
    # plt.show()
