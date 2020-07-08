import numpy as np
import matplotlib.pyplot as plt
import pytest
import scipy
import scatlib
from scatlib.interpolation import interpolate

def test_interpolation():
    for rank in [2, 3, 4, 5, 6, 7]:
        for degree in [1, 2, 3]:
            if degree >= rank - 1:
                break
            sizes = np.random.randint(5, 10, rank)
            t = np.random.randn(*sizes)

            weights = np.ones(degree)
            indices = [np.random.randint(max(s - 1, 0)) for s in sizes[:degree]]

            r = interpolate(t, weights, indices)
            assert(np.all(np.isclose(r, t[tuple(indices)])))

            weights = np.zeros(degree)
            r = interpolate(t, weights, indices)
            assert(np.all(np.isclose(r, t[tuple([i + 1 for i in indices])])))
