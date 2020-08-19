"""
Tests for the ScatteringDataField class.
"""
import matplotlib.pyplot as plt
import sys
import os
import copy
import netCDF4
import numpy as np
import scipy as sp
import scipy.interpolate
from scatlib.scattering_data_field import ScatteringDataFieldGridded
from utils import harmonic_random_field

class TestDataRandom:
    """
    Test scattering data field implementation for totally random scattering
    data.
    """
    def setup_method(self):
        """
        Setup test data.
        """
        self.f_grid = np.logspace(9, 11, 11)
        self.t_grid = np.linspace(250, 300, 6)
        self.lon_inc = np.ones(1)
        self.lat_inc = np.ones(1)
        self.lon_scat = np.ones(1)
        self.lat_scat = np.linspace(0, np.pi, 40)
        self.data = np.zeros((self.f_grid.size,
                             self.t_grid.size,
                             self.lon_inc.size,
                             self.lat_inc.size,
                             self.lon_scat.size,
                             self.lat_scat.size,
                             6))
        for i_f in range (self.f_grid.size):
            for i_t in range(self.t_grid.size):
                for i_c in range(6):
                    z = harmonic_random_field(self.lat_scat.size, 1)
                    self.data[i_f, i_t, 0, 0, 0, :, i_c] = z

        self.scattering_data = ScatteringDataFieldGridded(self.f_grid,
                                                          self.t_grid,
                                                          self.lon_inc,
                                                          self.lat_inc,
                                                          self.lon_scat,
                                                          self.lat_scat,
                                                          self.data)

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

    def interpolate_angles(self, lat_scat):
        """
        Reference implementation for angle interpolation.
        """
        axes = [5, 1, 2, 3, 4, 0, 6]
        data_t = np.transpose(self.data, axes)
        interpolator = sp.interpolate.RegularGridInterpolator([self.lat_scat],
                                                              data_t)
        return interpolator(lat_scat).transpose(axes)

    def test_frequency_interpolation(self):
        """
        Compare frequency interpolation to reference implementation.
        """
        df = self.f_grid[-1] - self.f_grid[0]
        frequencies = self.f_grid[0] + df * np.random.rand(10)
        result = self.scattering_data.interpolate_frequency(frequencies)
        result_ref = self.interpolate_frequency(frequencies)
        assert np.all(np.isclose(result.get_data(), result_ref))

    def test_temperature_interpolation(self):
        """
        Compare temperature interpolation to reference implementation.
        """
        dt = self.t_grid[-1] - self.t_grid[0]
        temperatures = self.t_grid[0] + dt * np.random.rand(10)
        result = self.scattering_data.interpolate_temperature(temperatures)
        result_ref = self.interpolate_temperature(temperatures)
        assert np.all(np.isclose(result.get_data(), result_ref))

    def test_angle_interpolation(self):
        """
        Compare angle interpolation to reference implementation.
        """
        thetas = np.pi * np.random.rand(180)
        result = self.scattering_data.interpolate_angles(np.ones(1),
                                                         np.ones(1),
                                                         np.ones(1),
                                                         thetas)
        result_ref = self.interpolate_angles(thetas)
        assert np.all(np.isclose(result.get_data(), result_ref))

    def test_transform_spectral(self):
        m_max = 0
        n_lat_scat = self.lat_scat.size
        l_max = n_lat_scat - n_lat_scat % 2
        if l_max == n_lat_scat:
            l_max -= 2
        spectral_data = self.scattering_data.to_spectral(l_max, m_max)
        gridded_data = spectral_data.to_gridded()
        assert np.all(np.isclose(self.data,
                                 gridded_data.get_data()))


class TestDataAzimuthallyRandom:
    """
    Test scattering data field implementation for totally random scattering
    data.
    """
    def setup_method(self):
        """
        Setup test data.
        """
        self.f_grid = np.logspace(9, 11, 11)
        self.t_grid = np.linspace(250, 300, 6)
        self.lon_inc = np.ones(1)
        self.lat_inc = np.linspace(0, np.pi, 21)
        self.lon_scat = np.linspace(0, 2 * np.pi, 20)
        self.lat_scat = np.linspace(0, np.pi, 40)
        self.data = np.zeros((self.f_grid.size,
                             self.t_grid.size,
                             self.lon_inc.size,
                             self.lat_inc.size,
                             self.lon_scat.size,
                             self.lat_scat.size,
                             6))
        for i_f in range (self.f_grid.size):
            for i_t in range(self.t_grid.size):
                print(i_t)
                for i_li in range(self.lat_inc.size):
                    for i_c in range(6):
                        z = harmonic_random_field(self.lat_scat.size,
                                                  self.lon_scat.size,
                                                  1)
                        self.data[i_f, i_t, 0, i_li, :, :, i_c] = z
        self.scattering_data = ScatteringDataFieldGridded(self.f_grid,
                                                          self.t_grid,
                                                          self.lon_inc,
                                                          self.lat_inc,
                                                          self.lon_scat,
                                                          self.lat_scat,
                                                          self.data)

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

    def interpolate_angles(self, lat_inc_new, lon_scat_new, lat_scat_new):
        """
        Reference implementation for angle interpolation.
        """
        axes = [3, 4, 5, 0, 1, 2, 6]
        data_t = np.transpose(self.data, axes)
        dims_in = data_t.shape
        dims_out = list(dims_in)
        dims_out[0] = lat_inc_new.size
        dims_out[1] = lon_scat_new.size
        dims_out[2] = lat_scat_new.size

        interpolator = sp.interpolate.RegularGridInterpolator([self.lat_inc,
                                                               self.lon_scat,
                                                               self.lat_scat],
                                                              data_t)
        angles = np.meshgrid(lat_inc_new, lon_scat_new, lat_scat_new, indexing="ij")
        angles = np.stack([a.ravel() for a in angles], axis=-1)
        data_interp = interpolator(angles).reshape(dims_out).transpose(axes)
        return data_interp

    def test_frequency_interpolation(self):
        """
        Compare frequency interpolation to reference implementation.
        """
        df = self.f_grid[-1] - self.f_grid[0]
        frequencies = self.f_grid[0] + df * np.random.rand(10)
        result = self.scattering_data.interpolate_frequency(frequencies)
        result_ref = self.interpolate_frequency(frequencies)
        assert np.all(np.isclose(result.get_data(), result_ref))

    def test_temperature_interpolation(self):
        """
        Compare temperature interpolation to reference implementation.
        """
        dt = self.t_grid[-1] - self.t_grid[0]
        temperatures = self.t_grid[0] + dt * np.random.rand(10)
        print("scatlib")
        result = self.scattering_data.interpolate_temperature(temperatures)
        print("python")
        result_ref = self.interpolate_temperature(temperatures)
        print("done")
        assert np.all(np.isclose(result.get_data(), result_ref))

    def test_angle_interpolation(self):
        """
        Compare angle interpolation to reference implementation.
        """
        thetas_inc = np.pi * np.random.rand(21)
        phi_scat = 2 * np.pi * np.random.rand(21)
        thetas_scat = np.pi * np.random.rand(21)

        result = self.scattering_data.interpolate_angles(np.ones(1),
                                                         thetas_inc,
                                                         phi_scat,
                                                         thetas_scat)
        result_ref = self.interpolate_angles(thetas_inc, phi_scat, thetas_scat)
        assert np.all(np.isclose(result.get_data(), result_ref))

    def test_transform_spectral(self):
        """
        Test transformation to spectral format and back.
        """
        m_max = self.lon_scat.size // 2 - 1
        n_lat_scat = self.lat_scat.size
        l_max = n_lat_scat - n_lat_scat % 2
        if l_max == n_lat_scat:
            l_max -= 2
        spectral_data = self.scattering_data.to_spectral(l_max, m_max)
        gridded_data = spectral_data.to_gridded()
        dg = gridded_data.get_data()
        return self.data, dg
        assert np.all(np.isclose(self.data,
                                 gridded_data.get_data()))

t = TestDataAzimuthallyRandom()
t.setup_method()
d, dg = t.test_transform_spectral()
