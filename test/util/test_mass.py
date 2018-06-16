import numpy as np
import scipy.io
import os
from src.distance.z_norm_euclidean import z_norm_euclidean
from src.util.util import mass, moving_average, moving_std


class TestMass(object):
    def test_mass(self):
        window_size = 10
        time_series_length = 1000

        time_series = np.arange(time_series_length)
        query = np.arange(window_size)

        distance_profile = np.zeros((time_series_length - window_size + 1, ))
        for i in range(time_series_length - window_size + 1):
            distance_profile[i] = z_norm_euclidean(query, time_series[i:i + window_size])

        ma = moving_average(time_series, window_size)
        mstd = moving_std(time_series, ma, window_size)

        time_series_freq = np.fft.fft(np.append(time_series, np.zeros(window_size, )))
        mass_dp = mass(query, time_series_freq, time_series_length, ma, mstd, np.mean(query), np.std(query))

        assert np.allclose(distance_profile, mass_dp, atol=1e-04)

    def test_mass_random(self):
        window_size = 10
        time_series_length = 1000

        time_series = np.random.rand(time_series_length)
        query = np.random.rand(window_size)

        distance_profile = np.zeros((time_series_length - window_size + 1, ))
        for i in range(time_series_length - window_size + 1):
            distance_profile[i] = z_norm_euclidean(query, time_series[i:i + window_size])

        ma = moving_average(time_series, window_size)
        mstd = moving_std(time_series, ma, window_size)

        time_series_freq = np.fft.fft(np.append(time_series, np.zeros(window_size, )))
        mass_dp = mass(query, time_series_freq, time_series_length, ma, mstd, np.mean(query), np.std(query))

        assert np.allclose(distance_profile, mass_dp, atol=1e-04)

    def test_mass_45713(self):
        path = os.path.abspath(os.path.dirname(__file__))
        data = scipy.io.loadmat(os.path.join(path, '../matrix_profile/test_data/penguin'))['penguin_sample']
        mp_45713_mass = scipy.io.loadmat(os.path.join(path, '../matrix_profile/test_data/mp_45713_mass'))['mp_45713_mass']
        mp_45713_mass = mp_45713_mass

        window_size = 800
        query = data[45712: 45712+800]

        ma = moving_average(data, window_size)
        mstd = moving_std(data, ma, window_size)

        distance_profile = np.zeros((data.shape[0] - window_size + 1, ))
        for i in range(data.shape[0] - window_size + 1):
            distance_profile[i] = z_norm_euclidean(query, data[i:i + window_size])

        data_freq = np.fft.fft(np.append(data, np.zeros(window_size, )))
        mass_dp = mass(query, data_freq, data.shape[0], ma, mstd, np.mean(query), np.std(query))

        assert np.allclose(mp_45713_mass, mass_dp.reshape(mp_45713_mass.shape), atol=1e-04)


