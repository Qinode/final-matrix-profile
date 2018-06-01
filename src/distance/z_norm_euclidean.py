import numpy as np


def z_norm_euclidean(series1, series2):
    assert series1.shape == series2.shape

    m1 = np.mean(series1)
    m2 = np.mean(series2)

    std1 = np.std(series1)
    std2 = np.std(series2)

    if std1 != 0:
        norm_1 = (series1 - m1 * np.ones(series1.shape))/std1
    else:
        norm_1 = np.zeros(series1.shape)

    if std2 != 0:
        norm_2 = (series2 - m2 * np.ones(series2.shape))/std2
    else:
        norm_2 = np.zeros(series2.shape)

    return np.linalg.norm(norm_1 - norm_2)