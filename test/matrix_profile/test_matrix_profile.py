import scipy.io
import numpy as np
from src.matrix_profile.matrixprofile import marix_profile


class TestMP(object):
    def test_mp(self):
        data = scipy.io.loadmat('test_data/penguin_160')['penguin_160']
        mp_test = scipy.io.loadmat('test_data/mp_160.mat')['matrixProfile_160']
        mpi_test = scipy.io.loadmat('test_data/mpi_160.mat')['profileIndex_160'] - np.ones(mp_test.shape)

        mp, mpi = marix_profile(data, data, 8)
        np.set_printoptions(precision=5)
        assert np.allclose(mp, mp_test)
        assert np.array_equal(mpi, mpi_test)

