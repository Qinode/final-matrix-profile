import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    f1 = pd.read_csv('./csv/09-01/f1s09-01.csv')
    time = pd.read_csv('./csv/09-01/times09-01.csv')

    f1 = f1.set_index('data name')
    time = time.set_index('data name')

    datasets = os.listdir('result')

    for data in datasets:
        if data not in ['TwoLeadECG', 'FaceFour', 'Wine']:
            continue

        print(data)
        print('mp', f1.at[data, 'mp best f1'], f1.at[data, 'mp best bit precision'], f1.at[data, 'mp best bit recall'],
              f1.at[data, 'mp best bit patterns'], time.at[data, 'mp best bit picking time'])
        print('gmp', f1.at[data, 'gmp best f1'], f1.at[data, 'gmp best bit precision'], f1.at[data, 'gmp best bit recall'],
              f1.at[data, 'gmp best bit patterns'], time.at[data, 'gmp best bit picking time'])
        print('ann', f1.at[data, 'ann best f1'], f1.at[data, 'ann best bit precision'], f1.at[data, 'ann best bit recall'],
              f1.at[data, 'ann best bit patterns'], time.at[data, 'ann best bit picking time'])


    mp_bit_table = {}
    gmp_bit_table = {}
    ann_bit_table = {}

    for data in datasets:
        mp_bit_table[data] = f1.at[data, 'mp best bit']
        gmp_bit_table[data] = f1.at[data, 'gmp best bit']
        ann_bit_table[data] = f1.at[data, 'ann best bit']
