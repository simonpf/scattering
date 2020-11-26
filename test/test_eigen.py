"""
Test of functions defined in eigen.h.
"""
import itertools
import numpy as np
import pytest
from scattering.eigen import (tensor_index,
                              get_submatrix,
                              get_subvector)


def test_indexing():
    """
    Tests indexing of sub-tensors in tensor.
    """
    for _ in range(100):
        rank = np.random.randint(2, 4)
        n_indices = np.random.randint(1, rank)
        sizes = [np.random.randint(5, 10) for i in range(rank)]
        index = [np.random.randint(0, s) for s in sizes[:n_indices]]
        data = np.random.rand(*sizes)
        print(index)
        e_ref = data[tuple(index)]
        e = tensor_index(data, index)
        print(e_ref)
        print(e)
        assert np.all(np.isclose(e_ref, e))

def test_indexing_scalar():
    """
    Tests indexing with full index array so that result is scalar.
    """
    for _ in range(100):
        rank = np.random.randint(2, 4)
        sizes = [np.random.randint(5, 10) for i in range(rank)]
        index = [np.random.randint(0, s) for s in sizes]
        data = np.random.rand(*sizes)
        e_ref = data[tuple(index)]
        e = tensor_index(data, index)
        assert np.all(np.isclose(e_ref, e))

def test_submatrix():
    """
    Tests extracting a submatrix from a rank-5 tensor.
    """
    rank = 5
    for _ in range(100):
        sizes = [np.random.randint(5, 10) for i in range(rank)]
        data = np.random.rand(*sizes)
        indices = [np.random.randint(0, s) for s in sizes]
        sub_indices = [indices[i] for i in [0, 2, 4]]
        m = get_submatrix(data, sub_indices)
        m_ref = data[indices[0], :, indices[2], :, indices[4]]
        assert np.all(np.isclose(m, m_ref))

def test_subvector():
    """
    Tests extracting subvector in a rank-5 tensor.
    """
    rank = 5
    for _ in range(100):
        sizes = [np.random.randint(5, 10) for i in range(rank)]
        data = np.random.rand(*sizes)
        indices = [np.random.randint(0, s) for s in sizes]
        sub_indices = [indices[i] for i in [0, 1, 2, 4]]
        m = get_subvector(data, sub_indices)
        m_ref = data[indices[0], indices[1], indices[2], :, indices[4]]
        assert np.all(np.isclose(m, m_ref))

