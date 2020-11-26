"""
Tests the classes of ScatteringDataField.h for scattering data from
spherical particles.
"""
import numpy as np
import scipy as sp
import scipy.interpolate
from utils import ScatteringDataBase
from scattering.scattering_data_field import ScatteringDataFieldGridded

class ScatteringDataSpherical(ScatteringDataBase):
    """
    Random spherical scattering data.
    """
    def __init__(self):
        self.f_grid = np.logspace(9, 11, 11)
        self.t_grid = np.linspace(250, 300, 6)
        self.lon_inc = np.ones(1)
        self.lat_inc = np.ones(1)
        self.lon_scat = np.ones(1)
        self.lat_scat = np.ones(1)
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
        self.scattering_data_spectral = self.scattering_data.to_spectral()
        self.scattering_data_fully_spectral = self.scattering_data_spectral.to_fully_spectral()

class TestScatteringDataFieldSpherical:
    def setup_method(self):
        self.data = ScatteringDataSpherical()

    def test_frequency_interpolation(self):
        df = self.data.f_grid[-1] - self.data.f_grid[0]
        frequencies = self.data.f_grid[0] + df * np.random.rand(10)
        reference = self.data.interpolate_frequency(frequencies)
        gridded = self.data.scattering_data.interpolate_frequency(frequencies)
        spectral = self.data.scattering_data_spectral.interpolate_frequency(frequencies)
        fully_spectral = self.data.scattering_data_fully_spectral.interpolate_frequency(frequencies)

        assert np.all(np.isclose(reference, gridded.get_data()))
        assert np.all(np.isclose(reference, spectral.to_gridded().get_data()))
        assert np.all(np.isclose(reference, fully_spectral.to_spectral().to_gridded().get_data()))

    def test_temperature_interpolation(self):
        dt = self.data.t_grid[-1] - self.data.t_grid[0]
        temperatures = self.data.t_grid[0] + dt * np.random.rand(10)
        reference = self.data.interpolate_temperature(temperatures)
        gridded = self.data.scattering_data.interpolate_temperature(temperatures)
        spectral = self.data.scattering_data_spectral.interpolate_temperature(temperatures)
        fully_spectral = self.data.scattering_data_fully_spectral.interpolate_temperature(temperatures)

        assert np.all(np.isclose(reference, gridded.get_data()))
        assert np.all(np.isclose(reference, spectral.to_gridded().get_data()))
        assert np.all(np.isclose(reference, fully_spectral.to_spectral().to_gridded().get_data()))

    def test_angle_interpolation(self):
        lon_inc = np.random.rand(10)
        lat_inc = np.random.rand(20)
        lon_scat = np.random.rand(30)
        lat_scat = np.random.rand(40)

        shape = (self.data.f_grid.size,
                 self.data.t_grid.size,
                 lon_inc.size,
                 lat_inc.size,
                 lon_scat.size,
                 lat_scat.size,
                 self.data.data.shape[-1])
        reference = np.broadcast_to(self.data.data, shape)
        gridded = self.data.scattering_data.interpolate_angles(lon_inc, lat_inc, lon_scat, lat_scat)

        spectral = self.data.scattering_data_spectral.to_gridded()
        spectral = spectral.interpolate_angles(lon_inc, lat_inc, lon_scat, lat_scat)

        fully_spectral = self.data.scattering_data_fully_spectral.to_spectral().to_gridded()
        fully_spectral = fully_spectral.interpolate_angles(lon_inc, lat_inc, lon_scat, lat_scat)

        assert np.all(np.isclose(reference, gridded.get_data()))
        assert np.all(np.isclose(reference, spectral.get_data()))
        assert np.all(np.isclose(reference, fully_spectral.get_data()))

    def test_addition(self):
        sum_1 = (self.data.scattering_data
                 + self.data.scattering_data
                 + self.data.scattering_data)
        sum_2 = (self.data.scattering_data_spectral
                 + self.data.scattering_data_spectral
                 + self.data.scattering_data_spectral)
        sum_3 = (self.data.scattering_data_fully_spectral
                 + self.data.scattering_data_fully_spectral
                 + self.data.scattering_data_fully_spectral)

        assert np.all(np.isclose(sum_1.get_data(),
                                 sum_2.to_gridded().get_data()))

        assert np.all(np.isclose(sum_2.get_data(),
                                 sum_3.to_spectral().get_data()))

        assert np.all(np.isclose(sum_1.get_data(),
                                 sum_3.to_spectral().to_gridded().get_data()))

    def test_scaling(self):
        scaled_1 = self.data.scattering_data * np.pi
        scaled_2 = self.data.scattering_data_spectral * np.pi
        scaled_3 = self.data.scattering_data_fully_spectral * np.pi

        assert np.all(np.isclose(scaled_1.get_data(),
                                 scaled_2.to_gridded().get_data()))

        assert np.all(np.isclose(scaled_2.get_data(),
                                 scaled_3.to_spectral().get_data()))

        assert np.all(np.isclose(scaled_1.get_data(),
                                 scaled_3.to_spectral().to_gridded().get_data()))
