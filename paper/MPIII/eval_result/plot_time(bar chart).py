import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import numpy as np

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%f' % float(height),
                ha='center', va='bottom')

if __name__ == '__main__':

    # (Uniform, Gaussian)
    picking_time = (13.03, 8.93, 0.08)
    label = ('Uniform', 'Gaussian', 'ANN')
    color = ('c', 'm', 'r')

    fig, ax = plt.subplots()

    opacity = 1

    for i in range(len(label) - 1):
        rects1 = ax.bar(i, picking_time[i],
                        alpha=opacity, width=0.3, label=label[i],
                        color=color[i])
        autolabel(rects1)

    ax.set_ylabel('Time (Second)')
    ax.set_xticks(range(len(label)))
    ax.set_xticklabels(('Uniform', 'Gaussian'))
    ax.legend()


    fig.tight_layout()
    plt.show()
