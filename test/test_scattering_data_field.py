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
        self.lat_scat = np.linspace(0, np.pi, 181)
        self.data = np.random.rand(6, 1, 1, 1, self.lat_scat.size)

        self.scattering_data = ScatteringDataFieldGridded(np.ones(1),
                                                          np.ones(1),
                                                          np.ones(1),
                                                          self.lat_scat,
                                                          self.data)

    def interpolate(self, angles):
        if len(angles.shape) == 1:
            angles = angles.reshape(-1, 1)
        if angles.shape[1] == 1:
            interpolator = sp.interpolate.RegularGridInterpolator([self.lat_scat],
                                                                  self.data[:, 0, 0, 0, :].T)
            return interpolator(angles)

    def test_interpolation(self):
        thetas = np.random.rand(1000)
        scattering_data = self.scattering_data
        result = scattering_data.interpolate(thetas)
        result_ref = self.interpolate(thetas)
        assert np.all(np.isclose(result, result_ref))

class TestDataAzimuthallyRandom:
    """
    Test scattering data field implementation for azimuthally random scattering
    data.
    """
    def setup_method(self):
        self.lat_inc = np.linspace(0, np.pi, 90)
        self.lon_scat = np.linspace(0, 2 * np.pi, 100)
        self.lat_scat = np.linspace(0, np.pi, 181)
        self.data = np.random.rand(6, 1, self.lat_inc.size, self.lon_scat.size, self.lat_scat.size)
        self.scattering_data = ScatteringDataFieldGridded(np.ones(1),
                                                          self.lat_inc,
                                                          self.lon_scat,
                                                          self.lat_scat,
                                                          self.data)

    def interpolate(self, angles):
        y = np.transpose(self.data[:, 0, :, :, :], axes=[1, 2, 3, 0])
        interpolator = sp.interpolate.RegularGridInterpolator([self.lat_inc,
                                                               self.lon_scat,
                                                               self.lat_scat],
                                                              y)
        return interpolator(angles)

    def test_interpolation(self):
        thetas = np.random.rand(1000, 3)
        scattering_data = self.scattering_data
        result = scattering_data.interpolate(thetas)
        result_ref = self.interpolate(thetas)
        assert np.all(np.isclose(result, result_ref))

class TestDataOriented:
    """
    Test scattering data field implementation for oriented scattering
    data.
    """
    def setup_method(self):
        self.lon_inc = np.linspace(0, 2 * np.pi, 50)
        self.lat_inc = np.linspace(0, np.pi, 90)
        self.lon_scat = np.linspace(0, 2 * np.pi, 100)
        self.lat_scat = np.linspace(0, np.pi, 181)
        self.data = np.random.rand(6, self.lon_inc.size, self.lat_inc.size, self.lon_scat.size, self.lat_scat.size)
        self.scattering_data = ScatteringDataFieldGridded(self.lon_inc,
                                                          self.lat_inc,
                                                          self.lon_scat,
                                                          self.lat_scat,
                                                          self.data)

    def interpolate(self, angles):
        y = np.transpose(self.data, axes=[1, 2, 3, 4, 0])
        interpolator = sp.interpolate.RegularGridInterpolator([self.lon_inc,
                                                               self.lat_inc,
                                                               self.lon_scat,
                                                               self.lat_scat],
                                                              y)
        return interpolator(angles)

    def test_interpolation(self):
        thetas = np.random.rand(1000, 4)
        thetas[:, 0] *= 2 * np.pi
        thetas[:, 1] *= np.pi
        thetas[:, 2] *= 2 * np.pi
        thetas[:, 3] *= np.pi

        scattering_data = self.scattering_data
        result = scattering_data.interpolate(thetas)
        result_ref = self.interpolate(thetas)
        assert np.all(np.isclose(result, result_ref))
