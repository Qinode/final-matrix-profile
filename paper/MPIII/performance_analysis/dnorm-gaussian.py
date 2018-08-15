import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def plot_bar(score1, score2, label1, label2, path):
    n_groups = len(score1)

    # means_men = (20, 35, 30, 35, 27)
    # std_men = (2, 3, 4, 1, 2)

    # means_women = (25, 32, 34, 20, 25)
    # std_women = (3, 5, 2, 3, 3)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, score1, bar_width,
                    alpha=opacity, color='b',
                    # yerr=std_men, error_kw=error_config,
                    label=label1)

    rects2 = ax.bar(index + bar_width, score2, bar_width,
                    alpha=opacity, color='r',
                    # yerr=std_women, error_kw=error_config,
                    label=label2)

    ax.set_xlabel('Performance Metrics')
    ax.set_ylabel('Scores')
    # ax.set_title('')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('F1 Score', 'Precision', 'Recall', 'Time'))
    ax.legend()

    fig.tight_layout()
    plt.savefig('{}/bar'.format(path))
    plt.close()

if __name__ == '__main__':
    result = pickle.load(open('result_dict.pkl', 'rb'))
    datasets = os.listdir('../eval_data')

    bits_range = [3, 4, 5, 6, 7]

    plot_dir = './dnorm-gaussian/'
    os.makedirs(plot_dir)

    for dataset in datasets:
        name = dataset[:-4]

        result_dict = {}
        quantization_way = ['gaussian', 'dnorm']

        print('===================================================================================================')
        for w in quantization_way:
            precisions = []
            recalls = []
            for bit in bits_range:
                save_data_path = '../eval_fig/normal-1-0/{}/saved-data/{}-{}bits'.format(name, w, bit)

                try:
                    matfile = sio.loadmat(save_data_path)
                except FileNotFoundError:
                    continue

                precisions.append(0 if matfile['precisions'].shape[0] == 0 else matfile['precisions'][0][-1])
                recalls.append(0 if matfile['recalls'].shape[0] == 0 else matfile['recalls'][0][-1])

            precisions = np.array(precisions)
            recalls = np.array(recalls)
            f1 = (np.divide(1.0, precisions, out=np.zeros_like(precisions), where=precisions != 0) + np.divide(
                1.0, recalls, out=np.zeros_like(recalls), where=recalls != 0))
            f1 = np.divide(2.0, f1, out=np.zeros_like(f1), where=f1 != 0)

            print('{} - {}'.format(name, w))
            print('Precisions: {}'.format(precisions))
            print('recalls   : {}'.format(recalls))
            print('F1        : {}'.format(f1))

    print('===================================================================================================')

