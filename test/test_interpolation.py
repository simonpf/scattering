import itertools
import numpy as np
import matplotlib.pyplot as plt
import pytest
import scipy as sp
import scipy.interpolate
import scatlib
from scatlib.interpolation import (interpolate,
                                   RegularGridInterpolator,
                                   downsample_dimension)
#
# Interpolation
#

def test_interpolation_with_given_weights():
    for rank in [2, 3, 4, 5, 6, 7]:
        for degree in [1, 2, 3]:
            if degree >= rank - 1:
                break
            sizes = np.random.randint(5, 10, rank)
            t = np.random.randn(*sizes)

            weights = np.ones(degree)
            indices = np.array([np.random.randint(max(s - 1, 0)) for s in sizes[:degree]])
            r = interpolate(t, weights, indices)
            assert(np.all(np.isclose(r, t[tuple(indices)])))

            weights = np.zeros(degree)
            r = interpolate(t, weights, indices)
            assert(np.all(np.isclose(r, t[tuple([i + 1 for i in indices])])))

def test_interpolation():
    rank = 5
    degree = 3

    sizes = np.random.randint(4, 10, rank)
    t = np.random.randn(*sizes)
    grids = [np.arange(sizes[i]) for i in range(degree)]

    positions = [g[:-1] + np.random.uniform(size=g.size - 1) for g in grids]
    positions = list(itertools.product(*positions))

    sp_interpolator = sp.interpolate.RegularGridInterpolator(grids, t)
    sp_results = sp_interpolator(positions)

    interpolator = RegularGridInterpolator(grids)
    results = interpolator.interpolate(t, positions)

    assert(np.all(np.isclose(results, sp_results)))

def test_interpolation_with_precalculated_weights():
    rank = 5
    degree = 3

    sizes = np.random.randint(4, 10, rank)
    t = np.random.randn(*sizes)
    grids = [np.arange(sizes[i]) for i in range(degree)]

    positions = [g[:-1] + np.random.uniform(size=g.size - 1) for g in grids]
    positions = list(itertools.product(*positions))

    sp_interpolator = sp.interpolate.RegularGridInterpolator(grids, t)
    sp_results = sp_interpolator(positions)

    interpolator = RegularGridInterpolator(grids)
    weights = interpolator.calculate_weights(positions)
    results = interpolator.interpolate(t, weights)

    assert(np.all(np.isclose(results, sp_results)))

def test_interpolation_degenerate_dimensions():
    rank = 5
    degree = 3

    interpolation_axis = np.random.choice(list(range(3)))
    sizes = [1, 1, 1] + list(np.random.randint(4, 10, rank - 3))
    sizes[interpolation_axis] = np.random.randint(4, 10)

    t = np.random.randn(*sizes)
    grids = [np.arange(sizes[i]) for i in range(degree)]

    positions = [[0]] * 3
    g = grids[interpolation_axis]
    positions[interpolation_axis] = g[:-1] + np.random.uniform(size=g.size - 1)
    positions = np.array(list(itertools.product(*positions)))

    sp_interpolator = sp.interpolate.RegularGridInterpolator([grids[interpolation_axis]], np.squeeze(t))
    print(positions, grids)
    sp_results = sp_interpolator(positions[:, interpolation_axis])

    interpolator = RegularGridInterpolator(grids)
    results = interpolator.interpolate(t, positions)

    assert(np.all(np.isclose(results, sp_results)))

def test_downsampling():
    x = np.linspace(0, 2 * np.pi, 1001)
    y = np.broadcast_to(np.sin(x).reshape(1, -1), (2000, 1001))
    y_down = downsample_dimension(y, x, np.array([0.5 * np.pi, 1.5 * np.pi]), 0., 2.0 * np.pi)
    assert np.all(np.isclose(y_down[:, 0], 2.0 / np.pi))
    assert np.all(np.isclose(y_down[:, 1], -2.0 / np.pi))



