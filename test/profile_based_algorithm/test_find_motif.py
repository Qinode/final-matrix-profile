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

        result = find_motif(mp, mpi, window_size, penguin, k=2, R=[2, 2])
        motif_pair_0, neighbor_0 = result[0]
        motif_pair_1, neighbor_1 = result[1]

        groud_truth_pair_0, groud_truth_neighbor_0 = np.array([6062, 31329]), np.array([73168, 29234, 108643])
        groud_truth_pair_1, groud_truth_neighbor_1 = np.array([6030, 103190]), np.array([31300, 73143, 24590, 59546, 29208])
        assert np.allclose(motif_pair_0, groud_truth_pair_0)
        assert np.allclose(groud_truth_neighbor_0, neighbor_0)
        assert np.allclose(motif_pair_1, groud_truth_pair_1)
        assert np.allclose(groud_truth_neighbor_1, neighbor_1)
