"""
Test utils providing acces to test data.
"""
import os
import numpy as np
import scipy
from scipy.special import roots_legendre, sph_harm

SCATLIB_TEST_PATH = "@SCATLIB_TEST_PATH@"

def get_data_azimuthally_random():
    return os.path.join(SCATLIB_TEST_PATH, "data", "scattering_data_azimuthally_random.nc")


def harmonic_random_field(n_lat, n_lon, n_components=10):
    """
    Generates a harmonic random field of by mixing a given number
    of random SH modes with Gaussian weights.

    Args:
        n_lat(int): The size of the latitude grid
        n_lon(int): The size of the longitude grid
        n_components: How many SHT components to mix (degault: 10)
    Returns:
        2D array of size (n_lon, n_lat) containing the random
        harmonic field.
    """
    l_max = n_lat - n_lat % 1
    if l_max == n_lat:
        l_max -= 1
    m_max = max(min(n_lon // 2 - 1, l_max), 0)
    lat_grid, _ = np.sort(np.arccos(roots_legendre(n_lat)))
    lon_grid = np.linspace(0, 2.0 * np.pi, n_lon + 1)[:-1]
    data = np.zeros((lon_grid.size, lat_grid.size))
    for _ in range(n_components):
        l = np.random.randint(0, l_max)
        if (m_max == 0) or (l == 0):
            m = 0
        else:
            m = np.random.randint(-min(l, m_max), min(l, m_max))

        xx, yy = np.meshgrid(lat_grid, lon_grid, indexing="xy")
        data += np.random.rand() * sph_harm(m, l, yy, xx).real
    return data
