"""
Tests the ScatteringData C++ class using azimuthally random scattering data.
"""
import sys
import os
import netCDF4
import numpy as np
import scipy as sp
import scipy.interpolate
from scatlib.scattering_data import ScatteringDataGridded

# Import test utils.
try:
    sys.path.append(os.path.dirname(__file__))
except Exception:
    pass

import utils

class ScatteringDataAzymuthallyRandom:
    """
    Reference implementation for handling of azimuthally random scattering
    data.
    """
    def __init__(self, filename):
        """
        Load scattering data from NetCDF4 file.

        Args:
            filename: The path of the file to load.
        """
        handle = netCDF4.Dataset(filename)
        self.azimuth_angles_incoming = handle["azimuth_angles_incoming"][:].data
        self.zenith_angles_incoming = handle["zenith_angles_incoming"][:].data
        self.azimuth_angles_scattering = handle["azimuth_angles_scattering"][:].data
        self.zenith_angles_scattering = handle["zenith_angles_scattering"][:].data
        self.phase_matrix = handle["phase_matrix"][:].data
        self.extinction_matrix = handle["extinction_matrix"][:].data
        self.absorption_vector = handle["absorption_vector"][:].data
        self.backscattering_coeff = np.zeros((0, 0))
        self.forwardscattering_coeff = np.zeros((0, 0))

    def get_data(self):
        """
        Data arguments to requires to create a ScatteringDataGridded object.

        Returns:
            tuple containing the arguments required to created a
            scatlib.ScatteringDataGridded object.
        """
        return [self.azimuth_angles_incoming,
                self.zenith_angles_incoming,
                self.azimuth_angles_scattering,
                self.zenith_angles_scattering,
                self.phase_matrix,
                self.extinction_matrix,
                self.absorption_vector,
                self.backscattering_coeff,
                self.forwardscattering_coeff]

    def interpolate_phase_matrix(self, angles):
        """
        Interpolates phase matrix to given scattering angles.

        Args:
            angles: 1D array of scattering angles of length n.

        Returns:
            [n x 6] matrix containing the phase matrix elements interpolated
            to the given angles.
        """
        grid = self.zenith_angles_scattering.ravel()
        data = np.squeeze(self.phase_matrix[:, ..., :].T)
        interpolator = sp.interpolate.RegularGridInterpolator([grid], data)
        return interpolator(angles)

    def interpolate_extinction_matrix(self, angles):
        return np.ones(angles.shape[0]) * self.extinction_matrix[0, 0, 0]

class TestScatteringDataGridded:

    def setup_method(self):
        self.scattering_data = ScatteringDataAzymuthallyRandom(utils.get_data_azimuthally_random())
        self.scattering_data_gridded = ScatteringDataGridded(*self.scattering_data.get_data())

    def test_phase_matrix_interpolation(self):
        """
        Compares results of phase matrix interpolation from reference implementation
        and C++ implementation.
        """
        n = 1000
        thetas = 180 * np.random.uniform(size=n)
        phase_matrix_ref = self.scattering_data.interpolate_phase_matrix(thetas)
        phase_matrix = self.scattering_data_gridded.get_phase_matrix(thetas)
        assert np.all(np.isclose(phase_matrix, phase_matrix_ref))

    def test_phase_matrix_interpolation_3_angles(self):
        """
        Same as above but uses 3-dimensional input for C++ function.
        """
        n = 1000
        thetas = 180 * np.random.uniform(size=(n, 3))
        phase_matrix = self.scattering_data_gridded.get_phase_matrix(thetas)
        phase_matrix_ref = self.scattering_data.interpolate_phase_matrix(thetas[:, -1])
        assert np.all(np.isclose(phase_matrix, phase_matrix_ref))

    def test_phase_matrix_interpolation_4_angles(self):
        """
        Same as above but uses 4-dimensional input for C++ function.
        """
        n = 1000
        thetas = 180 * np.random.uniform(size=(n, 4))
        phase_matrix = self.scattering_data_gridded.get_phase_matrix(thetas)
        phase_matrix_ref = self.scattering_data.interpolate_phase_matrix(thetas[:, -1])
        assert np.all(np.isclose(phase_matrix, phase_matrix_ref))

    def test_extinction_matrix_interpolation_0_angels(self):
        """
        Same as above but uses no input to C++ function because for totally
        random data the value is constant.
        """
        extinction_ref = self.scattering_data.extinction_matrix[0, 0, 0]
        extinction = self.scattering_data_gridded.get_extinction_matrix()
        assert np.isclose(extinction, extinction_ref)

    def test_extinction_matrix_interpolation(self):
        """
        Compares results of extinction matrix interpolation from reference implementation
        and C++ implementation.
        """
        n = 1000
        thetas = 180 * np.random.uniform(size=n)
        extinction_ref = self.scattering_data.interpolate_extinction_matrix(thetas)
        extinction = self.scattering_data_gridded.get_extinction_matrix(thetas)
        assert np.all(np.isclose(extinction, extinction_ref))

    def test_extinction_matrix_interpolation_2_angles(self):
        """
        Compares results of extinction matrix interpolation from reference implementation
        and C++ implementation.
        """
        n = 1000
        thetas = 180 * np.random.uniform(size=(n, 2))
        extinction_ref = self.scattering_data.interpolate_extinction_matrix(thetas[:, 1])
        extinction = self.scattering_data_gridded.get_extinction_matrix(thetas)
        assert np.all(np.isclose(extinction, extinction_ref))
