import itertools
import numpy as np
import matplotlib.pyplot as plt
import pytest
import scipy as sp
import scipy.interpolate
import scatlib
from scatlib.interpolation import interpolate, RegularGridInterpolator, RegularRegridder

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


#
# Regridding
#

def setup_grids(rank, n_dims, low=2, high=10):
    sizes = np.random.randint(low, high, rank)
    dimensions = np.random.choice(range(rank), n_dims, replace=False)
    old_grids = [np.arange(sizes[d]) for d in dimensions]
    new_grids = [np.arange(np.random.randint(1, sizes[d])) for d in dimensions]
    t = np.random.randn(*sizes)

    return old_grids, new_grids, dimensions, t

def regrid_scipy(old_grids, new_grids, dimensions, t):
    grids = [np.arange(s) for s in t.shape]
    output_grids = [np.arange(s) for s in t.shape]
    for i, d in enumerate(dimensions):
        output_grids[d] = new_grids[i]
    positions = np.meshgrid(*output_grids, indexing="ij")
    positions = np.stack([p.ravel() for p in positions], axis=-1)

    interpolator = sp.interpolate.RegularGridInterpolator(grids, t)
    results = interpolator(positions)
    results = results.reshape(tuple([o.size for o in output_grids]))
    return results

def regrid_scatlib(old_grids, new_grids, dimensions, t):
    regridder = RegularRegridder(old_grids, new_grids, dimensions)
    result = regridder.regrid(t)
    return result

def test_regrid():
    rank = 4
    n_dims = 2

    for i in range(10):
        old_grids, new_grids, dimensions, t = setup_grids(rank, n_dims, 4, 10)
        results_sp = regrid_scipy(old_grids, new_grids, dimensions, t)
        results = regrid_scatlib(old_grids, new_grids, dimensions, t)
        assert(np.all(np.isclose(results_sp, results)))
