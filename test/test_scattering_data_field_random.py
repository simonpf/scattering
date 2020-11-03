"""
Tests for the ScatteringDataField classes.
"""

import numpy as np
import scipy as sp
from scipy.special import roots_legendre
from utils import (harmonic_random_field, ScatteringDataBase, get_latitude_grid)
from scatlib.scattering_data_field import (ScatteringDataFieldGridded,
                                           ScatteringDataFieldSpectral,
                                           ScatteringDataFieldFullySpectral)


class ScatteringDataRandom(ScatteringDataBase):
    """
    Test data emulating data from randomly oriented particles.
    """
    def __init__(self):
        """
        Generates random test data.
        """
        self.f_grid = np.logspace(9, 11, 11)
        self.t_grid = np.linspace(250, 300, 6)
        self.lon_inc = np.ones(1)
        self.lat_inc = np.ones(1)
        self.lon_scat = np.ones(1)
        self.lat_scat = get_latitude_grid(180)
        self.data = np.ones((self.f_grid.size,
                             self.t_grid.size,
                             self.lon_inc.size,
                             self.lat_inc.size,
                             self.lon_scat.size,
                             self.lat_scat.size,
                             6))
        for i_f in range(self.f_grid.size):
            for i_t in range(self.t_grid.size):
                for i_c in range(6):
                    z = harmonic_random_field(1, self.lat_scat.size)
                    self.data[i_f, i_t, 0, 0, 0, :, i_c] = z
                    self.data[i_f, i_t, 0, 0, 0, :, i_c] += max(np.random.rand(),
                                                                0.1)

        self.scattering_data = ScatteringDataFieldGridded(self.f_grid,
                                                          self.t_grid,
                                                          self.lon_inc,
                                                          self.lat_inc,
                                                          self.lon_scat,
                                                          self.lat_scat,
                                                          self.data)
        self.scattering_data_spectral = self.scattering_data.to_spectral()
        self.sht_scat = self.scattering_data_spectral.get_sht_scat()
        l = self.sht_scat.get_l_max()
        m = self.sht_scat.get_m_max()
        self.scattering_data_spectral_2 = self.scattering_data.to_spectral(l - 2, m)
        self.scattering_data_fully_spectral = self.scattering_data_spectral.to_fully_spectral()
        self.sht_inc = self.scattering_data_fully_spectral.get_sht_inc()

    def interpolate_angles(self, lon_inc_new, lat_inc_new, lon_scat_new, lat_scat_new):
        """
        Reference implementation for angle interpolation.
        """
        axes = [5, 1, 2, 3, 4, 0, 6]
        data_t = np.transpose(self.data, axes)

        interpolator = sp.interpolate.RegularGridInterpolator([self.lat_scat],
                                                              data_t)
        data_interp = interpolator(lat_scat_new.reshape(-1, 1)).transpose(axes)

        dims_in = data_interp.shape
        dims_out = list(dims_in)
        dims_out[2] = lon_inc_new.size
        dims_out[3] = lat_inc_new.size
        dims_out[4] = lon_scat_new.size
        dims_out[5] = lat_scat_new.size
        return np.broadcast_to(data_interp, dims_out)

    def integrate_scattering_angles(self):
        """Numerical integration over scattering angles using trapezoidal rule. """
        return 2 * np.pi * np.trapz(self.data, x=-np.cos(self.lat_scat), axis=5)[..., 0, :]

class TestScatteringDataFieldRandom:
    """
    Tests the class ScatteringDataFieldGridded, ScatteringDataFieldSpectral and
    ScatteringDataFieldFullySpectral with randomly-oriented scattering data.
    """
    def setup_method(self):
        self.data = ScatteringDataRandom()

    def test_frequency_interpolation(self):
        """
        Frequency interpolation is tested for all data formats by comparison
        with reference implementation.
        """
        df = self.data.f_grid[-1] - self.data.f_grid[0]
        frequencies = self.data.f_grid[0] + df * np.random.rand(10)
        reference = self.data.interpolate_frequency(frequencies)
        gridded = self.data.scattering_data.interpolate_frequency(frequencies)
        spectral = self.data.scattering_data_spectral.interpolate_frequency(frequencies)
        spectral_2 = self.data.scattering_data_spectral_2.interpolate_frequency(frequencies)
        fully_spectral = self.data.scattering_data_fully_spectral.interpolate_frequency(frequencies)

        assert np.all(np.isclose(reference, gridded.get_data()))
        assert np.all(np.isclose(reference, spectral.to_gridded().get_data()))
        assert np.all(np.isclose(reference, spectral_2.to_gridded().get_data()))
        assert np.all(np.isclose(reference, fully_spectral.to_spectral().to_gridded().get_data()))

    def test_temperature_interpolation(self):
        """
        Temperature interpolation is tested for all data formats by comparison
        with reference implementation.
        """
        dt = self.data.t_grid[-1] - self.data.t_grid[0]
        temperatures = self.data.t_grid[0] + dt * np.random.rand(10)
        reference = self.data.interpolate_temperature(temperatures)
        gridded = self.data.scattering_data.interpolate_temperature(temperatures)
        spectral = self.data.scattering_data_spectral.interpolate_temperature(temperatures)
        spectral_2 = self.data.scattering_data_spectral_2.interpolate_temperature(temperatures)
        fully_spectral = self.data.scattering_data_fully_spectral.interpolate_temperature(temperatures)

        assert np.all(np.isclose(reference, gridded.get_data()))
        assert np.all(np.isclose(reference, spectral.to_gridded().get_data()))
        assert np.all(np.isclose(reference, spectral_2.to_gridded().get_data()))
        assert np.all(np.isclose(reference, fully_spectral.to_spectral().to_gridded().get_data()))

    def test_angle_interpolation(self):
        """
        Angle interpolation is tested for all data formats by converting them to gridded
        format and then performing the angle interpolation.
        """
        lon_inc = np.ones(10)
        lat_inc = np.ones(20)
        lon_scat = np.ones(30)
        lat_scat = np.linspace(0, np.pi, 42)[1:-1]

        shape = (self.data.f_grid.size,
                 self.data.t_grid.size,
                 lon_inc.size,
                 lat_inc.size,
                 lon_scat.size,
                 lat_scat.size,
                 self.data.data.shape[-1])
        reference = self.data.interpolate_angles(lon_inc, lat_inc, lon_scat, lat_scat)
        gridded = self.data.scattering_data.interpolate_angles(lon_inc, lat_inc, lon_scat, lat_scat)

        spectral = self.data.scattering_data_spectral.to_gridded()
        spectral = spectral.interpolate_angles(lon_inc, lat_inc, lon_scat, lat_scat)

        fully_spectral = self.data.scattering_data_fully_spectral.to_spectral().to_gridded()
        fully_spectral = fully_spectral.interpolate_angles(lon_inc, lat_inc, lon_scat, lat_scat)

        return reference, gridded, spectral, fully_spectral
        assert np.all(np.isclose(reference, gridded.get_data()))
        assert np.all(np.isclose(reference, spectral.get_data()))
        assert np.all(np.isclose(reference, fully_spectral.get_data()))

    def test_addition(self):
        """
        Addition of scattering data fields is tested for all formats and checked for
        consistency.
        """
        sum_1 = (self.data.scattering_data
                 + self.data.scattering_data
                 + self.data.scattering_data)
        sum_2 = (self.data.scattering_data_spectral
                 + self.data.scattering_data_spectral
                 + self.data.scattering_data_spectral)
        sum_3 = (self.data.scattering_data_spectral_2
                 + self.data.scattering_data_spectral_2
                 + self.data.scattering_data_spectral_2)
        sum_4 = (self.data.scattering_data_fully_spectral
                 + self.data.scattering_data_fully_spectral
                 + self.data.scattering_data_fully_spectral)

        assert np.all(np.isclose(sum_1.get_data(),
                                 sum_2.to_gridded().get_data()))

        assert np.all(np.isclose(sum_1.get_data(),
                                 sum_3.to_gridded().get_data()))

        assert np.all(np.isclose(sum_2.get_data(),
                                 sum_4.to_spectral().get_data()))

        assert np.all(np.isclose(sum_1.get_data(),
                                 sum_4.to_spectral().to_gridded().get_data()))

    def test_scaling(self):
        """
        Scaling of scattering data fields is tested for all formats and checked
        for consistency.
        """
        scaled_1 = self.data.scattering_data * np.pi
        scaled_2 = self.data.scattering_data_spectral * np.pi
        scaled_3 = self.data.scattering_data_spectral_2 * np.pi
        scaled_4 = self.data.scattering_data_fully_spectral * np.pi

        assert np.all(np.isclose(scaled_1.get_data(),
                                 scaled_2.to_gridded().get_data()))

        assert np.all(np.isclose(scaled_1.get_data(),
                                 scaled_3.to_gridded().get_data()))

        assert np.all(np.isclose(scaled_2.get_data(),
                                 scaled_4.to_spectral().get_data()))

        assert np.all(np.isclose(scaled_1.get_data(),
                                 scaled_4.to_spectral().to_gridded().get_data()))

    def test_set_data(self):
        """
        Setting of data for given temperature and frequency indices is
        tested for all formats.
        """
        result_gridded = ScatteringDataFieldGridded(self.data.f_grid,
                                                    self.data.t_grid,
                                                    self.data.lon_inc,
                                                    self.data.lat_inc,
                                                    self.data.lon_scat,
                                                    self.data.lat_scat,
                                                    self.data.data.shape[-1])

        result_spectral = ScatteringDataFieldSpectral(self.data.f_grid,
                                                      self.data.t_grid,
                                                      self.data.lon_inc,
                                                      self.data.lat_inc,
                                                      self.data.sht_scat,
                                                      self.data.data.shape[-1])

        result_spectral = ScatteringDataFieldSpectral(self.data.f_grid,
                                                      self.data.t_grid,
                                                      self.data.lon_inc,
                                                      self.data.lat_inc,
                                                      self.data.sht_scat,
                                                      self.data.data.shape[-1])

        result_fully_spectral = ScatteringDataFieldFullySpectral(self.data.f_grid,
                                                                 self.data.t_grid,
                                                                 self.data.sht_inc,
                                                                 self.data.sht_scat,
                                                                 self.data.data.shape[-1])

        for i, f in enumerate(self.data.f_grid):
            for j, t in enumerate(self.data.t_grid):
                d = self.data.scattering_data.interpolate_frequency([f])
                d = d.interpolate_temperature([t])
                result_gridded.set_data(i, j, d)

                d = self.data.scattering_data_spectral.interpolate_frequency([f])
                d = d.interpolate_temperature([t])
                result_spectral.set_data(i, j, d)

                d = self.data.scattering_data_fully_spectral.interpolate_frequency([f])
                d = d.interpolate_temperature([t])
                result_fully_spectral.set_data(i, j, d)


        assert np.all(np.isclose(self.data.scattering_data.get_data(),
                                 result_gridded.get_data()))
        assert np.all(np.isclose(self.data.scattering_data_spectral.get_data(),
                                 result_spectral.get_data()))
        assert np.all(np.isclose(self.data.scattering_data_fully_spectral.get_data(),
                                 result_fully_spectral.get_data()))

    def test_integration(self):
        """
        Check consistency of integration functions for gridded and spectral format
        and compare to reference implementation using numpy.
        """
        i_ref = self.data.integrate_scattering_angles()
        i1 = self.data.scattering_data.integrate_scattering_angles()
        i2 = self.data.scattering_data_spectral.integrate_scattering_angles()
        assert np.all(np.isclose(i1, i_ref, 1e-1))
        assert np.all(np.isclose(i2, i_ref, 1e-1))

    def test_normalization(self):
        """
        Check that scattering-angle integrals of normalized fields correspond
        to normalization value.
        """
        data_gridded = self.data.scattering_data.copy()
        data_spectral = self.data.scattering_data_spectral.copy()
        data_gridded.normalize(4.0 * np.pi)
        data_spectral.normalize(4.0 * np.pi)

        i1 = data_gridded.integrate_scattering_angles()
        i2 = data_spectral.integrate_scattering_angles()
        i3 = data_gridded.to_spectral().integrate_scattering_angles()
        i4 = data_spectral.to_gridded().integrate_scattering_angles()

        assert np.all(np.isclose(i1[..., 0], 4.0 * np.pi))
        assert np.all(np.isclose(i2[..., 0], 4.0 * np.pi))
        assert np.all(np.isclose(i3[..., 0], 4.0 * np.pi))
        assert np.all(np.isclose(i4[..., 0], 4.0 * np.pi))

    def test_set_n_scattering_coeffs(self):
        """
        Ensure that reduction of scattering coefficients works by reducing to 1 component
        and comparing with original data.
        """
        data_gridded = self.data.scattering_data.copy()
        data_spectral = self.data.scattering_data_spectral.copy()
        data_fully_spectral = self.data.scattering_data_fully_spectral.copy()

        data_gridded.set_number_of_scattering_coeffs(1)
        data_spectral.set_number_of_scattering_coeffs(1)
        data_fully_spectral.set_number_of_scattering_coeffs(1)

        assert np.all(np.isclose(data_gridded.get_data()[..., 0],
                                 self.data.scattering_data.get_data()[..., 0]))
        assert np.all(np.isclose(data_spectral.get_data()[..., 0],
                                 self.data.scattering_data_spectral.get_data()[..., 0]))
        assert np.all(np.isclose(data_fully_spectral.get_data()[..., 0],
                                 self.data.scattering_data_fully_spectral.get_data()[..., 0]))

    def test_downsampling(self):
        """
        Check consistency of integration functions for gridded and spectral format
        and compare to reference implementation using numpy.
        """
        dummy_grid = np.array([np.pi])
        data_downsampled_gridded = self.data.scattering_data.downsample_scattering_angles(dummy_grid,
                                                                                          dummy_grid)
        i_ref = self.data.scattering_data.integrate_scattering_angles()
        i_1 = data_downsampled_gridded.integrate_scattering_angles()
        assert np.all(np.isclose(i_ref, i_1))
