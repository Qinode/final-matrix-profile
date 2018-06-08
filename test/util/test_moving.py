import numpy as np
from src.util.util import moving_average, mean_std, moving_std

class TestMoving(object):

    def test_moving_average(self):
        window_size = 10
        series_size = 1000

        series = np.arange(series_size)

        ma = moving_average(series, window_size)
        ma_test, _ = mean_std(window_size, series)

        assert np.allclose(ma, ma_test)

    def test_moving_std(self):
        window_size = 10
        series_size = 1000

        series = np.arange(series_size)

        ma = moving_average(series, window_size)
        mstd = moving_std(series, ma, window_size)
        ma_test, mstd_test = mean_std(window_size, series)

        assert np.allclose(ma, ma_test)
        assert np.allclose(mstd, mstd_test)

    def test_moving_random(self):
        window_size = 10
        series_size = 1000

        series = np.random.rand(series_size)

        ma = moving_average(series, window_size)
        mstd = moving_std(series, ma, window_size)
        ma_test, mstd_test = mean_std(window_size, series)

        assert np.allclose(ma, ma_test)
        assert np.allclose(mstd, mstd_test)
