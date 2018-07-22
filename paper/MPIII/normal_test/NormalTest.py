import matplotlib.pyplot as plt
import scipy.io
import scipy.stats
import statsmodels.api as sm
import numpy as np
import os


def normal_test(dataset, data_name, gaussian):
    path = os.path.dirname((os.path.abspath(__file__)))

    mat_file = scipy.io.loadmat(dataset)
    data = mat_file['data'].squeeze()
    mean, std = np.mean(data), np.std(data)
    z_data = (data - mean)/std

    probplot = sm.ProbPlot(z_data)
    probplot.qqplot(line='45')
    # scipy.stats.probplot(z_data, plot=plt)

    if gaussian:
        plt.savefig(os.path.join(path, 'normal-test-result/gaussian/{}.png'.format(data_name)))
        plt.clf()
    else:
        plt.savefig(os.path.join(path, 'normal-test-result/dnorm/{}.png'.format(data_name)))
        plt.clf()


if __name__ == '__main__':
    gaussian = ['Adiac', 'BirdChicken', 'Cricket_Y', 'DistalPhalanxOutlineAgeGroup'
                , 'ECGFiveDays', 'FISH', 'Herring', 'ItalyPowerDemand', 'Lighting7', 'MedicalImages'
                , 'MiddlePhalanxOutlineAgeGroup', 'MoteStrain', 'Plane', 'ProximalPhalanxOutlineCorrect'
                , 'ProximalPhalanxTW', 'SwedishLeaf', 'Trace', 'TwoLeadECG', 'Worms', 'WormsTwoClass']


    path = os.path.dirname(os.path.abspath(__file__))
    eval_path = os.path.join(path, '../eval')
    data = os.listdir(eval_path)
    for d in data:
        if d[:-4] in gaussian:
            normal_test(os.path.join(eval_path, d), d[:-4], True)
        else:
            normal_test(os.path.join(eval_path, d), d[:-4], False)
