import scipy.io
import matplotlib.pyplot as plt
import scipy.stats

from paper.MPIII.visual import *

if __name__ == '__main__':

    arrow_head = scipy.io.loadmat('ArrowHead')
    data = arrow_head['data']

    mp = arrow_head['matrixProfile']
    mpi = arrow_head['profileIndex'] - 1
    tp = arrow_head['labIdx'] - 1

    p = 0.2

    window_size = arrow_head['subLen'][0][0]
    bits = 5

    choice = np.random.choice(mp.shape[0], 10)

    for i in choice:
        print('{}/{}'.format(i, mp.shape[0]))
        sub = data[i: i+window_size]
        sub = sub - np.mean(sub)
        sub = sub / np.std(sub)
        sub = sub.reshape(window_size, )
        scipy.stats.probplot(sub, plot=plt)
    plt.show()

    nsample = np.random.normal(loc=0, scale=1, size=window_size)
    scipy.stats.probplot(nsample, plot=plt)

    plt.show()

    plt.plot(nsample)
    plt.show()

