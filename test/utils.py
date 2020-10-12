"""
Test utils providing acces to test data.
"""
import os
import numpy as np
import scipy as sp
import scipy.interpolate
from scipy.special import roots_legendre, sph_harm
from reference import ParticleFile

SCATLIB_TEST_PATH = "@SCATLIB_TEST_PATH@"

def get_data_azimuthally_random():
    return os.path.join(SCATLIB_TEST_PATH, "data", "scattering_data_azimuthally_random.nc")

RANDOM_DATA_PATH = os.path.join(SCATLIB_TEST_PATH, "data", "random")
file_1 = os.path.join(RANDOM_DATA_PATH,
                      "Dmax00191um_Dveq00101um_Mass5.00440e-10kg.nc")
particle_random_1 = ParticleFile(file_1)
file_2 = os.path.join(RANDOM_DATA_PATH,
                      "Dmax00191um_Dveq00101um_Mass5.00440e-10kg.nc")
particle_random_2 = ParticleFile(file_2)

AZIMUTHALLY_RANDOM_DATA_PATH = os.path.join(SCATLIB_TEST_PATH, "data", "azimuthally_random")
file_1 = os.path.join(AZIMUTHALLY_RANDOM_DATA_PATH,
                      "Dmax00191um_Dveq00101um_Mass5.00440e-10kg.nc")
particle_azimuthally_random_1 = ParticleFile(file_1)
file_2 = os.path.join(AZIMUTHALLY_RANDOM_DATA_PATH,
                      "Dmax01014um_Dveq00770um_Mass2.19345e-07kg.nc")
particle_azimuthally_random_2 = ParticleFile(file_2)

def harmonic_random_field(n_lon, n_lat, n_components=10):
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
    l_max = max(n_lat // 2 - 3, 0)
    m_max = max(min(n_lon // 2 - 1, l_max), 0)
    lat_grid = np.linspace(0, np.pi, n_lat + 1)
    lat_grid = 0.5 * (lat_grid[1:] + lat_grid[:-1])
    lon_grid = np.linspace(0, 2.0 * np.pi, n_lon + 1)[:-1]
    data = np.zeros((lon_grid.size, lat_grid.size))
    for _ in range(n_components):
        l = np.random.randint(0, l_max)
        if (m_max == 0) or (l == 0):
            m = 0
        else:
            m = np.random.randint(-min(l, m_max), min(l, m_max))

        xx, yy = np.meshgrid(lat_grid, lon_grid, indexing="xy")
        data += 6.66 * sph_harm(m, l, yy, xx).real
    return data

def get_latitude_grid(n):
    """
    Return Gauss-Legendre latitude grid expected by SHTns.
    """
    return np.sort(np.arccos(roots_legendre(n)[0]))

class ScatteringDataBase:
    """
    Base class for scattering data object. Implements reference functions to test
    the ScatteringDataField classes.
    """
    def interpolate_frequency(self, frequencies):
        """
        Reference implementation for frequency interpolation.
        """
        interpolator = sp.interpolate.RegularGridInterpolator([self.f_grid],
                                                              self.data)
        return interpolator(frequencies)

    def interpolate_temperature(self, temperatures):
        """
        Reference implementation for temperature interpolation.
        """
        axes = [1, 0, 2, 3, 4, 5, 6]
        data_t = np.transpose(self.data, axes)
        interpolator = sp.interpolate.RegularGridInterpolator([self.t_grid],
                                                              data_t)
        return interpolator(temperatures).transpose(axes)

    def interpolate_angles(self, lon_inc_new, lat_inc_new, lon_scat_new, lat_scat_new):
        """
        Reference implementation for angle interpolation.
        """
        axes = [2, 3, 4, 5, 0, 1, 6]
        data_t = np.transpose(self.data, axes)
        dims_in = data_t.shape
        dims_out = list(dims_in)
        dims_out[0] = lon_inc_new.size
        dims_out[1] = lat_inc_new.size
        dims_out[2] = lon_scat_new.size
        dims_out[3] = lat_scat_new.size

        interpolator = sp.interpolate.RegularGridInterpolator([self.lon_inc,
                                                               self.lat_inc,
                                                               self.lon_scat,
                                                               self.lat_scat],
                                                              data_t)
        angles = np.meshgrid(lon_inc_new, lat_inc_new, lon_scat_new, lat_scat_new, indexing="ij")
        angles = np.stack([a.ravel() for a in angles], axis=-1)
        axes = [4, 5, 0, 1, 2, 3, 6]
        data_interp = interpolator(angles).reshape(dims_out).transpose(axes)
        return data_interp
