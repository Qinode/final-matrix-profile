import matplotlib.pyplot as plt
from random import shuffle
import scipy.io
import scipy.stats
import statsmodels.api as sm
import numpy as np
import os


def distribution_test(dataset, data_name, **kwargs):
    path = os.path.dirname((os.path.abspath(__file__)))

    mat_file = scipy.io.loadmat(dataset)
    data = mat_file['data'].squeeze()

    probplot = sm.ProbPlot(data, scipy.stats.uniform, fit=True)
    probplot.qqplot(line='45')

    plt.savefig(os.path.join(path, 'test_result/uniform/{}.png'.format(data_name)))
    plt.title(data_name)
    plt.clf()


if __name__ == '__main__':
    # gaussian = ['Adiac', 'BirdChicken', 'DistalPhalanxOutlineAgeGroup'
    #             , 'ECGFiveDays', 'FISH', 'Herring', 'ItalyPowerDemand', 'Lighting7', 'MedicalImages'
    #             , 'MiddlePhalanxOutlineAgeGroup', 'MoteStrain', 'Plane', 'ProximalPhalanxOutlineCorrect'
    #             , 'ProximalPhalanxTW', 'SwedishLeaf', 'Trace', 'TwoLeadECG', 'Worms', 'WormsTwoClass']

    path = os.path.dirname(os.path.abspath(__file__))
    eval_path = os.path.join(path, '../eval_data')
    data = os.listdir(eval_path)
    for d in data:
        print(d)
        distribution_test(os.path.join(eval_path, d), d[:-4])
