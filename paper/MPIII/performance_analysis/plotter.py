import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_bar(score1, score2, label1, label2, path):
    n_groups = 3

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
    ax.set_xticklabels(('F1 Score', 'Precision', 'Recall'))
    ax.legend()

    fig.tight_layout()
    plt.savefig('{}/bar'.format(path))
    plt.close()

if __name__ == '__main__':
    result = pickle.load(open('./pickles/result_dict08-19.pkl', 'rb'))
    datasets = os.listdir('../eval_data')

    bits_range = [3, 4, 5, 6, 7]
    cats = ['WithoutANN', 'ANN']

    for dataset in datasets:
        name = dataset[:-4]

        for bit in bits_range:
            plot_dir = './plots08-19/{}/{}'.format(name, bit)
            os.makedirs(plot_dir)

            # f1, precision, recall
            data = [[], []]

            key_error = False
            for i in range(len(cats)):
                try:
                    precisions = result[name][bit][cats[i]]['precisions']
                    recalls = result[name][bit][cats[i]]['recalls']
                except KeyError as error:
                    key_error = True
                    continue

                precision = 0 if len(precisions) == 0 else precisions[0][-1]
                recall = 0 if len(recalls) == 0 else recalls[0][-1]
                f1 = 0 if (precision == 0 or recall == 0) else np.around(2.0/((1.0/precision) + (1.0)/recall), decimals=3)
                time = result[name][bit][cats[i]]['time'][0][0]

                # data[i] = [f1, precision, recall, time]
                data[i] = [f1, precision, recall]
            if key_error:
                continue

            # scale time in (0, 1) range
            print('{} - {}'.format(name, bit))

            # total_time = data[0][3] + data[1][3]
            # data[0][3] = float(data[0][3])/total_time
            # data[1][3] = float(data[1][3])/total_time

            plot_bar(data[0], data[1], cats[0], cats[1], plot_dir)
