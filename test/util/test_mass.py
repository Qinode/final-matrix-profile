import numpy as np
from src.distance.z_norm_euclidean import z_norm_euclidean
from src.util.util import mass


class TestMass(object):

    def test_mass(self):
        window_size = 10
        time_series_length = 1000

        time_series = np.arange(time_series_length)
        query = np.arange(window_size)

        distance_profile = np.zeros((time_series_length - window_size + 1, ))
        for i in range(time_series_length - window_size + 1):
            distance_profile[i] = z_norm_euclidean(query, time_series[i:i + window_size])

        mass_dp = mass(query, time_series)

        assert np.allclose(distance_profile, mass_dp)
