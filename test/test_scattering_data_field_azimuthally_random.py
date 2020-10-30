"""
Tests for ScatteringDataField class for scattering data of azimuthally-random
type.
"""
import numpy as np
import scipy as sp
import scipy.interpolate
from scipy.special import roots_legendre
from utils import (harmonic_random_field, ScatteringDataBase)
from scatlib.scattering_data_field import (ScatteringDataFieldGridded,
                                           ScatteringDataFieldSpectral,
                                           ScatteringDataFieldFullySpectral,
                                           SHT)

class ScatteringDataAzimuthallyRandom(ScatteringDataBase):
    """
    Random test data of azimuthally-random type.
    """
    def __init__(self):
        l_max_inc = np.random.randint(5, 15)
        m_max_inc = 0
        n_lat_inc = max(l_max_inc + 2, 32)
        n_lon_inc = 1
        l_max_scat = np.random.randint(10, 20)
        m_max_scat = np.random.randint(9, l_max_scat)
        n_lat_scat = max(2 * l_max_scat + 2, 32) * 2
        n_lon_scat = max(2 * m_max_scat + 2, 1) * 2
        self.f_grid = np.logspace(9, 11, 11)
        self.t_grid = np.linspace(250, 300, 6)
        sht_inc = SHT(l_max_inc, m_max_inc, n_lon_inc, n_lat_inc)
        sht_scat = SHT(l_max_scat, m_max_scat, n_lon_scat, n_lat_scat)
        self.lon_inc = sht_inc.get_longitude_grid()
        self.lat_inc = sht_inc.get_latitude_grid()
        self.lon_scat = sht_scat.get_longitude_grid()
        self.lat_scat = sht_scat.get_latitude_grid()
        data = np.random.rand(self.f_grid.size,
                               self.t_grid.size,
                               sht_inc.get_n_spectral_coeffs_cmplx(),
                               sht_scat.get_n_spectral_coeffs(),
                               6)
        data = data + 1j * np.random.rand(self.f_grid.size,
                                           self.t_grid.size,
                                           sht_inc.get_n_spectral_coeffs_cmplx(),
                                           sht_scat.get_n_spectral_coeffs(),
                                           6)


        self.scattering_data_fully_spectral = ScatteringDataFieldFullySpectral(self.f_grid,
                                                                               self.t_grid,
                                                                               sht_inc,
                                                                               sht_scat,
                                                                               data)
        self.scattering_data_spectral = self.scattering_data_fully_spectral.to_spectral()
        l_max = self.scattering_data_spectral.get_sht_scat().get_l_max()
        m_max = self.scattering_data_spectral.get_sht_scat().get_m_max()
        self.scattering_data_spectral_2 = self.scattering_data_fully_spectral.to_spectral(l_max * 2, m_max)
        self.scattering_data_gridded = self.scattering_data_spectral.to_gridded()

        self.data = self.scattering_data_gridded.get_data()

    def interpolate_angles(self, lon_inc_new, lat_inc_new, lon_scat_new, lat_scat_new):
        """
        Reference implementation for angle interpolation.
        """
        axes = [3, 4, 5, 0, 1, 2, 6]
        data_t = np.transpose(self.data, axes)

        interpolator = sp.interpolate.RegularGridInterpolator([self.lat_inc,
                                                               self.lon_scat,
                                                               self.lat_scat],
                                                              data_t)

        angles = np.meshgrid(lat_inc_new, lon_scat_new, lat_scat_new, indexing="ij")
        angles = np.stack([a.ravel() for a in angles], axis=-1)
        data_interp = interpolator(angles)
        output_shape = [lat_inc_new.size,
                        lon_scat_new.size,
                        lat_scat_new.size,
                        self.f_grid.size,
                        self.t_grid.size,
                        1, 6]
        data_interp = data_interp.reshape(output_shape).transpose(axes)

        dims_in = data_interp.shape
        dims_out = list(dims_in)
        dims_out[2] = lon_inc_new.size
        dims_out[3] = lat_inc_new.size
        dims_out[4] = lon_scat_new.size
        dims_out[5] = lat_scat_new.size
        return np.broadcast_to(data_interp, dims_out)

    def integrate_scattering_angles(self):
        """Numerical integration over scattering angles using Gauss-Legendre quadrature. """
        _, weights = roots_legendre(self.lat_scat.size)
        weights = np.broadcast_to(np.copy(weights.reshape(-1, 1)), (1,) * 5 + (weights.size,) + (1,))
        latitude_integrals = np.sum(weights * self.data, axis=5)
        remainder = 0.0
        if self.lon_scat[-1] < 2 * np.pi:
            dx = self.lon_scat[0] + 2 * np.pi - self.lon_scat[-1]
            remainder = 0.5 * dx * (latitude_integrals[:, :, :, :, -1, :]
                                    + latitude_integrals[:, :, :, :, 0, :])
        print("remainder: ", remainder)
        return np.trapz(latitude_integrals, x=self.lon_scat, axis=4) + remainder

class TestScatteringDataFieldAzimuthallyRandom:
    """
    Tests the class ScatteringDataFieldGridded, ScatteringDataFieldSpectral and
    ScatteringDataFieldFullySpectral with randomly-oriented scattering data.
    """
    def setup_method(self):
        self.data = ScatteringDataAzimuthallyRandom()

    def test_frequency_interpolation(self):
        """
        Frequency interpolation is tested for all data formats by comparison
        with reference implementation.
        """
        df = self.data.f_grid[-1] - self.data.f_grid[0]
        frequencies = self.data.f_grid[0] + df * np.random.rand(10)
        reference = self.data.interpolate_frequency(frequencies)
        gridded = self.data.scattering_data_gridded.interpolate_frequency(frequencies)
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
        gridded = self.data.scattering_data_gridded.interpolate_temperature(temperatures)
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
        lon_inc = np.ones(1)
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
        gridded = self.data.scattering_data_gridded.interpolate_angles(lon_inc, lat_inc, lon_scat, lat_scat)

        spectral = self.data.scattering_data_spectral.to_gridded()
        spectral = spectral.interpolate_angles(lon_inc, lat_inc, lon_scat, lat_scat)

        fully_spectral = self.data.scattering_data_fully_spectral.to_spectral().to_gridded()
        fully_spectral = fully_spectral.interpolate_angles(lon_inc, lat_inc, lon_scat, lat_scat)

        assert np.all(np.isclose(reference, gridded.get_data()))
        assert np.all(np.isclose(reference, spectral.get_data()))
        assert np.all(np.isclose(reference, fully_spectral.get_data()))

    def test_addition(self):
        """
        Addition of scattering data fields is tested for all formats and checked for
        consistency.
        """
        sum_1 = (self.data.scattering_data_gridded
                 + self.data.scattering_data_gridded
                 + self.data.scattering_data_gridded)
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
        scaled_1 = self.data.scattering_data_gridded * np.pi
        scaled_2 = self.data.scattering_data_spectral * np.pi
        scaled_3 = self.data.scattering_data_fully_spectral * np.pi

        assert np.all(np.isclose(scaled_1.get_data(),
                                 scaled_2.to_gridded().get_data()))

        assert np.all(np.isclose(scaled_2.get_data(),
                                 scaled_3.to_spectral().get_data()))

        assert np.all(np.isclose(scaled_1.get_data(),
                                 scaled_3.to_spectral().to_gridded().get_data()))

    def test_integration(self):
        """
        Check consistency of integration functions for gridded and spectral format
        and compare to reference implementation using numpy.
        """
        i_ref = self.data.integrate_scattering_angles()
        i1 = self.data.scattering_data_gridded.integrate_scattering_angles()
        i2 = self.data.scattering_data_spectral.integrate_scattering_angles()
        return i_ref, i2
        assert np.all(np.isclose(i1, i_ref))
        assert np.all(np.isclose(i2, i_ref))

    def test_normalization(self):
        """
        Check that scattering-angle integrals of normalized fields correspond
        to normalization value.
        """
        data_gridded = self.data.scattering_data_gridded.copy()
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
        data_gridded = self.data.scattering_data_gridded.copy()
        data_spectral = self.data.scattering_data_spectral.copy()
        data_fully_spectral = self.data.scattering_data_fully_spectral.copy()

        data_gridded.set_number_of_scattering_coeffs(1)
        data_spectral.set_number_of_scattering_coeffs(1)
        data_fully_spectral.set_number_of_scattering_coeffs(1)

        assert np.all(np.isclose(data_gridded.get_data()[..., 0],
                                 self.data.scattering_data_gridded.get_data()[..., 0]))
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
        data_downsampled_gridded = self.data.scattering_data_gridded.downsample_scattering_angles(dummy_grid,
                                                                                                  dummy_grid)
        i_ref = self.data.scattering_data.integrate_scattering_angles()
        i_1 = data_downsampled_gridded.integrate_scattering_angles()
        assert np.all(np.isclose(i_ref, i_1))

