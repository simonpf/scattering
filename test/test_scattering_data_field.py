"""
Tests for the ScatteringDataField class.
"""
import sys
import os
import netCDF4
import numpy as np
import scipy as sp
import scipy.interpolate
from scatlib.scattering_data_field import ScatteringDataFieldGridded

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
        self.lat_scat = np.linspace(0, np.pi, 181)
        self.data = np.random.rand(self.f_grid.size,
                                   self.t_grid.size,
                                   self.lon_inc.size,
                                   self.lat_inc.size,
                                   self.lon_scat.size,
                                   self.lat_scat.size,
                                   6)
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
        result = self.scattering_data.interpolate_angles(np.ones(1), np.ones(1), np.ones(1), thetas)
        result_ref = self.interpolate_angles(thetas)
        assert np.all(np.isclose(result.get_data(), result_ref))


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
        self.lat_inc = np.linspace(0, np.pi, 91)
        self.lon_scat = np.linspace(0, 2 * np.pi, 91)
        self.lat_scat = np.linspace(0, np.pi, 181)
        self.data = np.random.rand(self.f_grid.size,
                                   self.t_grid.size,
                                   self.lon_inc.size,
                                   self.lat_inc.size,
                                   self.lon_scat.size,
                                   self.lat_scat.size,
                                   6)
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

    def interpolate_angles(self, angles):
        """
        Reference implementation for angle interpolation.
        """
        axes = [3, 4, 5, 0, 1, 2, 6]
        data_t = np.transpose(self.data, axes)
        interpolator = sp.interpolate.RegularGridInterpolator([self.lat_scat],
                                                              data_t)
        angles = np.meshgrid(angles[:, 0], angles[:, 1], angles[:, 2])
        angles = np.stack([a.ravel() for a in angles], axis=-1)
        return interpolator(angles).transpose(axes)

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
        thetas_inc = np.pi * np.random.rand(180)
        phi_inc = 2 * np.pi * np.random.rand(180)
        thetas_scat = np.pi * np.random.rand(180)
        angles = np.stack([thetas_inc, phi_inc, thetas_scat], axis=-1)

        result = self.scattering_data.interpolate_angles(thetas_inc,
                                                         phi_inc,
                                                         thetas_scat)
        result_ref = self.interpolate_angles(angles)
        assert np.all(np.isclose(result.get_data(), result_ref))
