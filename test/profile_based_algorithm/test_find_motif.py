import scipy.io
import numpy as np
import os
from src.profile_based_algorithm.find_motif import find_motif


class TestFindMotif(object):
    def test_find_motif(self):
        path = os.path.abspath(os.path.dirname(__file__))
        mp = scipy.io.loadmat(os.path.join(path, '../matrix_profile/test_data/py_mp_full.mat'))['mp']

        penguin = scipy.io.loadmat(os.path.join(path, '../matrix_profile/test_data/penguin.mat'))['penguin_sample']
        mpi = scipy.io.loadmat(os.path.join(path, '../matrix_profile/test_data/py_mpi_full.mat'))['mpi']

        mpi = mpi.reshape((109043, 1))
        mp = mp.reshape((109043, 1))

        window_size = 800

        penguin_freq = np.fft.fft(np.append(penguin, np.zeros((window_size, 1))))
        motif_pair, neighbor = find_motif(mp, mpi, window_size, penguin, penguin_freq)

        groud_truth_pair, groud_truth_neighbor = np.array([6062, 31329]), np.array([73168, 29234, 108643])
        assert np.allclose(motif_pair, groud_truth_pair)
        assert np.allclose(groud_truth_neighbor, neighbor)
