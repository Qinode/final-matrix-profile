import numpy as np
from src.util.util import moving_average, moving_std


class TestMoving(object):

    def mean_std(self, window_size, series):
        mean = np.zeros((series.shape[0] - window_size + 1))
        std = np.zeros((series.shape[0] - window_size + 1))

        for i in range(series.shape[0] - window_size + 1):
            mean[i] = np.mean(series[i: i+window_size])
            std[i] = np.std(series[i: i+window_size])

        return mean, std

    def test_moving(self):
        window_size = 10
        series_size = 1000

        series = np.arange(series_size)

        ma = moving_average(series, window_size)
        mstd = moving_std(series, ma, window_size)
        ma_test, mstd_test = self.mean_std(window_size, series)

        assert np.allclose(ma, ma_test)
        assert np.allclose(mstd, mstd_test)

    def test_moving_random(self):
        window_size = 10
        series_size = 1000

        series = np.random.rand(series_size)

        ma = moving_average(series, window_size)
        mstd = moving_std(series, ma, window_size)
        ma_test, mstd_test = self.mean_std(window_size, series)

        assert np.allclose(ma, ma_test)
        assert np.allclose(mstd, mstd_test)
