import numpy as np
from src.distance.z_norm_euclidean import z_norm_euclidean


class TestZNormDistance(object):
    def test_zero(self):
        s1 = np.array([1, 1, 1])
        s2 = np.array([1, 1, 1])
        d = z_norm_euclidean(s1, s2)
        assert d == 0

