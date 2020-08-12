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
        self.backscattering_coeff = np.ones((1, 1)) * self.phase_matrix[0, 0, 0, 0, 0]
        self.forwardscattering_coeff = np.ones((1, 1)) * self.phase_matrix[0, 0, 0, 0, 0]

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
        """
        Return extinction matrix for given scattering angles. Since
        the extinction matrix for azimuthally random data does not depend
        on the scattering angle, this function simply returns a vector
        of the same length as angles with the extinction matrix element
        (0, 0).

        Args:
            angles: N-element array with scattering angles. Since the
            extinction matrix is independent of the scattering angle the
            input is used only to determin the size of the output.

        Returns:
            Vector of the same length as the input with the single extinction
            matrix element.
        """
        return np.ones(angles.shape[0]) * self.extinction_matrix[0, 0, 0]

    def interpolate_absorption_vector(self, angles):
        """
        Work in the same way as interpolate_extinction_matrix but for the
        absorption vector.
        """
        return np.ones(angles.shape[0]) * self.absorption_vector[0, 0, 0]

    def interpolate_backscattering_coeff(self, angles):
        """
        Work in the same way as interpolate_absorption_vector but for the
        backscattering coefficient.
        """
        return np.ones(angles.shape[0]) * self.backscattering_coeff[0, 0]

    def interpolate_forwardscattering_coeff(self, angles):
        """
        Work in the same way as interpolate_absorption_vector but for the
        forwardscattering coefficient.
        """
        return np.ones(angles.shape[0]) * self.forwardscattering_coeff[0, 0]

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

    def test_absorption_vector_interpolation_0_angels(self):
        """
        Same as above but uses no input to C++ function because for totally
        random data the value is constant.
        """
        absorption_ref = self.scattering_data.absorption_vector[0, 0, 0]
        absorption = self.scattering_data_gridded.get_absorption_vector_data()
        assert np.isclose(absorption, absorption_ref)

    def test_absorption_vector_interpolation(self):
        """
        Compares results of absorption vector interpolation from reference implementation
        and C++ implementation.
        """
        n = 1000
        thetas = 180 * np.random.uniform(size=n)
        absorption_ref = self.scattering_data.interpolate_absorption_vector(thetas)
        absorption = self.scattering_data_gridded.get_absorption_vector(thetas)
        assert np.all(np.isclose(absorption, absorption_ref))

    def test_absorption_vector_interpolation_2_angles(self):
        """
        Compares results of absorption matrix interpolation from reference implementation
        and C++ implementation.
        """
        n = 1000
        thetas = 180 * np.random.uniform(size=(n, 2))
        absorption_ref = self.scattering_data.interpolate_absorption_vector(thetas[:, 1])
        absorption = self.scattering_data_gridded.get_absorption_vector(thetas)
        assert np.all(np.isclose(absorption, absorption_ref))

    def test_backscattering_coeff_interpolation_0_angels(self):
        """
        Same as above but uses no input to C++ function because for totally
        random data the value is constant.
        """
        backscattering_ref = self.scattering_data.backscattering_coeff[0, 0]
        backscattering = self.scattering_data_gridded.get_backscattering_coeff()
        print(backscattering)
        print(backscattering_ref)

        assert np.isclose(backscattering, backscattering_ref)

    def test_backscattering_coeff_interpolation(self):
        """
        Compares results of extinction matrix interpolation from reference implementation
        and C++ implementation.
        """
        n = 1000
        thetas = 180 * np.random.uniform(size=n)
        backscattering_ref = self.scattering_data.interpolate_backscattering_coeff(thetas)
        backscattering = self.scattering_data_gridded.get_backscattering_coeff(thetas)
        assert np.all(np.isclose(backscattering, backscattering_ref))

    def test_backscattering_coeff_interpolation_2_angles(self):
        """
        Compares results of extinction matrix interpolation from reference implementation
        and C++ implementation.
        """
        n = 1000
        thetas = 180 * np.random.uniform(size=(n, 2))
        backscattering_ref = self.scattering_data.interpolate_backscattering_coeff(thetas[:, 1])
        backscattering = self.scattering_data_gridded.get_backscattering_coeff(thetas)
        assert np.all(np.isclose(backscattering, backscattering_ref))

    def test_forwardscattering_coeff_interpolation_0_angels(self):
        """
        Same as above but uses no input to C++ function because for totally
        random data the value is constant.
        """
        forwardscattering_ref = self.scattering_data.forwardscattering_coeff[0, 0]
        forwardscattering = self.scattering_data_gridded.get_forwardscattering_coeff()
        print(forwardscattering)
        print(forwardscattering_ref)

        assert np.isclose(forwardscattering, forwardscattering_ref)

    def test_forwardscattering_coeff_interpolation(self):
        """
        Compares results of extinction matrix interpolation from reference implementation
        and C++ implementation.
        """
        n = 1000
        thetas = 180 * np.random.uniform(size=n)
        forwardscattering_ref = self.scattering_data.interpolate_forwardscattering_coeff(thetas)
        forwardscattering = self.scattering_data_gridded.get_forwardscattering_coeff(thetas)
        assert np.all(np.isclose(forwardscattering, forwardscattering_ref))

    def test_forwardscattering_coeff_interpolation_2_angles(self):
        """
        Compares results of extinction matrix interpolation from reference implementation
        and C++ implementation.
        """
        n = 1000
        thetas = 180 * np.random.uniform(size=(n, 2))
        forwardscattering_ref = self.scattering_data.interpolate_forwardscattering_coeff(thetas[:, 1])
        forwardscattering = self.scattering_data_gridded.get_forwardscattering_coeff(thetas)
 
