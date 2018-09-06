import pickle
import numpy as np
import os

if __name__ == '__main__':
    ann_result = pickle.load(open('pkls/ann_result.pkl', 'rb'))
    mp_result = pickle.load(open('pkls/mp_result.pkl', 'rb'))

    data_sets = os.listdir('result/')
    precisions = []
    recalls = []
    ptimes = []
    rtimes = []

    header = ['data name', 'mp best precision', 'ann best precision', 'mp best bit', 'ann best bit']
    precisions.append(header)
    recalls.append(header)

    for data in data_sets:
        pmp_idx = np.argmax(mp_result[data]['precisions'])
        pann_idx = np.argmax(ann_result[data]['precisions'])

        precisions.append([
            data,
            np.max(mp_result[data]['precisions']),
            np.max(ann_result[data]['precisions']),
            pmp_idx + 3,
            pann_idx + 3,
            mp_result[data]['patterns'][pmp_idx],
            ann_result[data]['patterns'][pann_idx]
        ])

        rmp_idx = np.argmax(mp_result[data]['recalls'])
        rann_idx = np.argmax(ann_result[data]['recalls'])
        recalls.append([
            data,
            np.max(mp_result[data]['recalls']),
            np.max(ann_result[data]['recalls']),
            rmp_idx + 3,
            pann_idx + 3,
            mp_result[data]['patterns'][rmp_idx],
            ann_result[data]['patterns'][rann_idx]
        ])

        ptimes.append([mp_result[data]['times'][pmp_idx], ann_result[data]['times'][pann_idx]])
        rtimes.append([mp_result[data]['times'][rmp_idx], ann_result[data]['times'][rann_idx]])

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    ptimes = np.array(ptimes)
    rtimes = np.array(rtimes)
