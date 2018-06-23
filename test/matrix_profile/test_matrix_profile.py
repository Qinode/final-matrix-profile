import scipy.io
import numpy as np
import os
from src.matrix_profile.matrixprofile import naive, stamp, stomp


class TestMP(object):
    def test_naive(self):
        path = os.path.abspath(os.path.dirname(__file__))

        data = scipy.io.loadmat(os.path.join(path, 'test_data/penguin_160'))['penguin_160']
        mp_test = scipy.io.loadmat(os.path.join(path, 'test_data/mp_160.mat'))['matrixProfile_160']
        mpi_test = scipy.io.loadmat(os.path.join(path, 'test_data/mpi_160.mat'))['profileIndex_160'] - np.ones(mp_test.shape)

        mp, mpi = naive(data, data, 8)
        assert np.allclose(mp, mp_test)
        assert np.array_equal(mpi, mpi_test)

    def test_naive_16000(self):
        path = os.path.abspath(os.path.dirname(__file__))

        data = scipy.io.loadmat(os.path.join(path, 'test_data/penguin_16000'))['penguin_16000']
        mp_test = scipy.io.loadmat(os.path.join(path, 'test_data/mp_16000.mat'))['matrixProfile_16000']
        mpi_test = scipy.io.loadmat(os.path.join(path, 'test_data/mpi_16000.mat'))['profileIndex_16000'] - np.ones(mp_test.shape)

        mp, mpi = naive(data, data, 8)
        assert np.allclose(mp, mp_test)
        assert np.array_equal(mpi, mpi_test)

    def test_stamp(self):
        path = os.path.abspath(os.path.dirname(__file__))

        data = scipy.io.loadmat(os.path.join(path, 'test_data/penguin_160'))['penguin_160']
        mp_test = scipy.io.loadmat(os.path.join(path, 'test_data/mp_160.mat'))['matrixProfile_160']
        mpi_test = scipy.io.loadmat(os.path.join(path, 'test_data/mpi_160.mat'))['profileIndex_160'] - np.ones(mp_test.shape)

        mp, mpi = stamp(data, data, 8, self_join=True)

        assert np.array_equal(mpi.reshape(mpi_test.shape), mpi_test)
        assert np.allclose(mp.reshape(mp_test.shape), mp_test, atol=1e-04)

    def test_stamp_16000(self):
        path = os.path.abspath(os.path.dirname(__file__))

        data = scipy.io.loadmat(os.path.join(path, 'test_data/penguin_16000'))['penguin_16000']
        mp_test = scipy.io.loadmat(os.path.join(path, 'test_data/mp_16000.mat'))['matrixProfile_16000']
        mpi_test = scipy.io.loadmat(os.path.join(path, 'test_data/mpi_16000.mat'))['profileIndex_16000'] - np.ones(mp_test.shape)

        mp, mpi = stamp(data, data, 800, self_join=True)
        
        assert np.array_equal(mpi.reshape(mpi_test.shape), mpi_test)
        assert np.allclose(mp.reshape(mp_test.shape), mp_test, atol=1e-04)

    def test_stamp_40000_70000(self):
        path = os.path.abspath(os.path.dirname(__file__))

        data = scipy.io.loadmat(os.path.join(path, 'test_data/penguin_40000_70000'))['penguin_40000_70000']
        mp_test = scipy.io.loadmat(os.path.join(path, 'test_data/mp_40000_70000.mat'))['matrixProfile40000_70000']
        mpi_test = scipy.io.loadmat(os.path.join(path, 'test_data/mpi_40000_70000.mat'))['profileIndex40000_70000'] - np.ones(mp_test.shape)
    
        mp, mpi = stamp(data, data, 800, self_join=True)
    
        assert np.array_equal(mpi.reshape(mpi_test.shape), mpi_test)
        assert np.allclose(mp.reshape(mp_test.shape), mp_test, atol=1e-04)

    def test_stamp_full(self):
        path = os.path.abspath(os.path.dirname(__file__))

        data = scipy.io.loadmat(os.path.join(path, 'test_data/penguin'))['penguin_sample']
        mp_test = scipy.io.loadmat(os.path.join(path, 'test_data/mp.mat'))['matrixProfile']
        mpi_test = scipy.io.loadmat(os.path.join(path, 'test_data/mpi.mat'))['profileIndex'] - np.ones(mp_test.shape)

        mp, mpi = stamp(data, data, 800, self_join=True)

        # scipy.io.savemat(os.path.join(path, 'test_data/py_mp_full.mat'), {'mp': mp})
        # scipy.io.savemat(os.path.join(path, 'test_data/py_mpi_full.mat'), {'mpi': mpi})

        assert np.allclose(np.around(mp.reshape(mp_test.shape), decimals=4), mp_test, atol=1e-04)
        assert np.array_equal(mpi.reshape(mpi_test.shape), mpi_test)

    def test_stomp(self):
        path = os.path.abspath(os.path.dirname(__file__))

        data = scipy.io.loadmat(os.path.join(path, 'test_data/penguin_160'))['penguin_160']
        mp_test = scipy.io.loadmat(os.path.join(path, 'test_data/mp_160.mat'))['matrixProfile_160']
        mpi_test = scipy.io.loadmat(os.path.join(path, 'test_data/mpi_160.mat'))['profileIndex_160'] - np.ones(mp_test.shape)

        mp, mpi = stomp(data, data, 8, self_join=True)

        assert np.array_equal(mpi.reshape(mpi_test.shape), mpi_test)
        assert np.allclose(mp.reshape(mp_test.shape), mp_test, atol=1e-04)

    def test_stomp_16000(self):
        path = os.path.abspath(os.path.dirname(__file__))

        data = scipy.io.loadmat(os.path.join(path, 'test_data/penguin_16000'))['penguin_16000']
        mp_test = scipy.io.loadmat(os.path.join(path, 'test_data/mp_16000.mat'))['matrixProfile_16000']
        mpi_test = scipy.io.loadmat(os.path.join(path, 'test_data/mpi_16000.mat'))['profileIndex_16000'] - np.ones(mp_test.shape)

        mp, mpi = stomp(data, data, 800, self_join=True)
        assert np.array_equal(mpi.reshape(mpi_test.shape), mpi_test)
        assert np.allclose(mp.reshape(mp_test.shape), mp_test, atol=1e-04)

    def test_stomp_full(self):
        path = os.path.abspath(os.path.dirname(__file__))

        data = scipy.io.loadmat(os.path.join(path, 'test_data/penguin'))['penguin_sample']
        mp_test = scipy.io.loadmat(os.path.join(path, 'test_data/py_mp_full.mat'))['mp']
        mpi_test = scipy.io.loadmat(os.path.join(path, 'test_data/py_mpi_full.mat'))['mpi']

        mp, mpi = stomp(data, data, 800, self_join=True)

        scipy.io.savemat(os.path.join(path, 'test_data/py_mp_full_stomp.mat'), {'mp': mp})
        scipy.io.savemat(os.path.join(path, 'test_data/py_mpi_full_stomp.mat'), {'mpi': mpi})

        assert np.allclose(np.around(mp.reshape(mp_test.shape), decimals=4), mp_test, atol=1e-04)
        assert np.array_equal(mpi.reshape(mpi_test.shape), mpi_test)
