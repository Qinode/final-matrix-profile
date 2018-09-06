import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import numpy as np

if __name__ == '__main__':

    n_groups = 3

    # F1, precision, recall
    means_uniform = (0.59, 0.58, 0.64)
    std_uniform = (0.26, 0.27, 0.27)

    means_gaussian = (0.56, 0.61, 0.58)
    std_gaussian = (0.30, 0.28, 0.34)

    means_ann = (0.59, 0.57, 0.66)
    std_ann = (0.25, 0.27, 0.25)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.25

    opacity = 0.8
    error_config = {'ecolor': '0.3'}
    color = ('c', 'm', 'r')
    rects1 = ax.bar(index, means_uniform, bar_width,
                    alpha=opacity, color='c',
                    yerr=std_uniform, error_kw=error_config,
                    label='Uniform')

    rects2 = ax.bar(index + bar_width, means_gaussian, bar_width,
                    alpha=opacity, color='m',
                    yerr=std_gaussian, error_kw=error_config,
                    label='Gaussian')

    # rects3 = ax.bar(index + 2 * bar_width, means_ann, bar_width,
    #                 alpha=opacity, color='r',
    #                 yerr=std_ann, error_kw=error_config,
    #                 label='ANN')

    ax.set_ylabel('Scores')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('F1', 'Precision', 'Recall'))
    ax.legend()

    fig.tight_layout()
    plt.show()

