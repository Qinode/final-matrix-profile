import scipy.io
import numpy as np
import os
from src.matrix_profile.matrixprofile import marix_profile


class TestMP(object):
    def test_mp(self):
        path = os.path.abspath(os.path.dirname(__file__))

        data = scipy.io.loadmat(os.path.join(path, 'test_data/penguin_160'))['penguin_160']
        mp_test = scipy.io.loadmat(os.path.join(path, 'test_data/mp_160.mat'))['matrixProfile_160']
        mpi_test = scipy.io.loadmat(os.path.join(path, 'test_data/mpi_160.mat'))['profileIndex_160'] - np.ones(mp_test.shape)

        mp, mpi = marix_profile(data, data, 8)
        assert np.allclose(mp, mp_test)
        assert np.array_equal(mpi, mpi_test)

