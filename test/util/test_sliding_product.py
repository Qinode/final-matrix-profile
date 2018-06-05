import numpy as np
from src.util.util import sliding_dot_product


class TestSlidingProduct(object):
    def test_sliding_product(self):
        time_series_size = 1000
        window_size = 10

        time_series = np.arange(time_series_size)
        query = np.arange(window_size)

        qt = sliding_dot_product(query, time_series)
        for i in range(time_series_size - window_size + 1):
            assert np.allclose(qt[i], query.transpose().dot(time_series[i:i+window_size]))


    def test_sliding_product_random(self):
        time_series_size = 1000
        window_size = 10

        time_series = np.random.rand(time_series_size)
        query = np.random.rand(window_size)

        qt = sliding_dot_product(query, time_series)
        for i in range(time_series_size - window_size + 1):
            assert np.allclose(qt[i], query.transpose().dot(time_series[i:i+window_size]))