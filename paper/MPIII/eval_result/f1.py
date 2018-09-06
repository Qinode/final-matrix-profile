import pickle
import numpy as np
import os


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


if __name__ == '__main__':
    ann_result = pickle.load(open('pkls/ann_result.pkl', 'rb'))
    mp_result = pickle.load(open('pkls/mp_result09-01.pkl', 'rb'))
    gmp_result = pickle.load(open('pkls/gmp_result09-01.pkl', 'rb'))

    data_sets = os.listdir('result/')
    f1s = []

    f1_header = ['data name', 'mp best f1', 'ann best f1', 'gmp best f1', 'mp best bit', 'ann best bit', 'gmp best bit',
              'mp best bit patterns', 'ann best bit patterns', 'gmp best bit patterns',
              'mp best bit precision', 'mp best bit recall', 'ann best bit precision', 'ann best bit recall',
              'gmp best bit precision', 'gmp best bit recall']
    f1s.append(f1_header)

    time_header = ['data name', 'mp best bit picking time', 'ann best bit picking time', 'gmp best bit picking time',
                   'mp best bit time', 'ann best bit time', 'gmp best bit time']
    ftimes = []
    ftimes.append(time_header)

    for data in data_sets:
        mp_precisions = mp_result[data]['precisions']
        mp_recalls = mp_result[data]['recalls']
        mp_f1 = (div0(1.0, mp_precisions) + div0(1.0, mp_recalls))
        mp_f1 = div0(2.0, mp_f1)

        gmp_precisions = gmp_result[data]['precisions']
        gmp_recalls = gmp_result[data]['recalls']
        gmp_f1 = (div0(1.0, gmp_precisions) + div0(1.0, gmp_recalls))
        gmp_f1 = div0(2.0, gmp_f1)

        ann_precisions = ann_result[data]['precisions']
        ann_recalls = ann_result[data]['recalls']
        ann_f1 = (div0(1.0, ann_precisions) + div0(1.0, ann_recalls))
        ann_f1 = div0(2.0, ann_f1)

        fmp_idx = np.argmax(mp_f1)
        fann_idx = np.argmax(ann_f1)
        fgmp_idx = np.argmax(gmp_f1)

        f1s.append([
            data,
            np.max(mp_f1),
            np.max(ann_f1),
            np.max(gmp_f1),
            fmp_idx + 3,
            fann_idx + 3,
            fgmp_idx + 3,
            mp_result[data]['patterns'][fmp_idx],
            ann_result[data]['patterns'][fann_idx],
            gmp_result[data]['patterns'][fgmp_idx],
            mp_precisions[fmp_idx],
            mp_recalls[fmp_idx],
            ann_precisions[fann_idx],
            ann_recalls[fann_idx],
            gmp_precisions[fgmp_idx],
            gmp_recalls[fgmp_idx]
        ])

        ftimes.append([data, mp_result[data]['picking_times'][fmp_idx], ann_result[data]['picking_times'][fann_idx],
                       gmp_result[data]['picking_times'][fgmp_idx], mp_result[data]['times'][fmp_idx],
                       ann_result[data]['times'][fann_idx],
                       gmp_result[data]['times'][fgmp_idx]])

    f1s = np.array(f1s)
    ftimes = np.array(ftimes)
