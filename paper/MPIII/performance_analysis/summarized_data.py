import os
import scipy.io as sio
import pickle

if __name__ == '__main__':
    datasets = os.listdir('../eval_data/')
    bits_range = [3, 4, 5, 6, 7]

    exp_result = {}

    for dataset in datasets:
        name = dataset[:-4]
        ann_dir = 'ann_selection08-17'
        norm_dir = 'eval-2(dnorm, gaussian)(3-7)'

        exp_result[name] = {}

        for bit in bits_range:
            norm_mat_path = '../eval_fig/{}/{}/saved-data/dnorm-{}bits.mat'.format(norm_dir, name, bit)
            ann_mat_path = '../eval_fig/{}/{}/saved-data/dnorm-{}bits.mat'.format(ann_dir, name, bit)

            temp = {}

            try:
                norm_mat = sio.loadmat(norm_mat_path)
            except FileNotFoundError as error:
                print(error)
                continue

            try:
                ann_mat = sio.loadmat(ann_mat_path)
            except FileNotFoundError as error:
                print(error)
                continue

            temp['ANN'] = {'precisions': ann_mat['precisions'], 'recalls': ann_mat['recalls'], 'time': ann_mat['process_time']}
            temp['WithoutANN'] = {'precisions': norm_mat['precisions'], 'recalls': norm_mat['recalls'], 'time': norm_mat['process_time']}

            exp_result[name][bit] = temp

    pickle.dump(exp_result, open('result_dict08-19.pkl', 'wb'))
