import pickle
import numpy as np
import os

if __name__ == '__main__':
    ann_result = pickle.load(open('pkls/ann_result.pkl', 'rb'))
    mp_result = pickle.load(open('pkls/mp_result.pkl', 'rb'))

    data_sets = os.listdir('result/')
    precisions = []
    recalls = []
    times = []

    header = ['data name', 'mp best precision', 'ann best precision', 'mp best bit', 'ann best bit']
    precisions.append(header)
    recalls.append(header)

    for data in data_sets:
        precisions.append([
            data,
            np.max(mp_result[data]['precisions']),
            np.max(ann_result[data]['precisions']),
            np.argmax(mp_result[data]['precisions']) + 3,
            np.argmax(ann_result[data]['precisions']) + 3
        ])

        recalls.append([
            data,
            np.max(mp_result[data]['recalls']),
            np.max(ann_result[data]['recalls']),
            np.argmax(mp_result[data]['recalls']) + 3,
            np.argmax(ann_result[data]['recalls']) + 3
        ])

    precisions = np.array(precisions)
    recalls = np.array(recalls)
