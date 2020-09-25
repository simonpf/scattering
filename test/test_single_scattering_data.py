"""
Tests the ScatteringData C++ class using azimuthally random scattering data.
"""
import sys
import os
import netCDF4
import numpy as np
import scipy as sp
import scipy.interpolate
from scatlib.single_scattering_data import (SingleScatteringData,
                                            ParticleType,
                                            SHT)

#Import test utils.
try:
    sys.path.append(os.path.dirname(__file__))
except Exception:
    pass

import utils

def particle_to_single_scattering_data_random(particle_data,
                                              frequency,
                                              temperature):
    """
    Convert ssdb.Particle to SingleScatteringData object.
    """
    f_grid = np.array([frequency])
    t_grid = np.array([temperature])

    pm = particle_data.phase_matrix[:, :, :, :, :, :-1, :]
    em = particle_data.extinction_matrix
    av = particle_data.absorption_vector
    bsc = particle_data.backward_scattering_coeff
    fsc = particle_data.forward_scattering_coeff

    sd = SingleScatteringData(f_grid,
                              t_grid,
                              particle_data.lon_inc,
                              particle_data.lat_inc,
                              particle_data.lon_scat,
                              particle_data.lat_scat[:-1],
                              pm,
                              em,
                              av,
                              bsc,
                              fsc)
    return sd

class TestSingleScatteringDataRandom:
    """
    Test loading and manipulation of gridded single scattering data.

    Attributes:
        f_grid (np.array): The frequency grid on which the scattering data
            is defined.
        t_grid (np.array): The temperature grid on which the scattering data
            is defined.
        data (SingleScatteringData): Single scattering data object containing
            the test data.
        particle(ssdb.Particle): Particle object providing access to the test
            data.
    """

    def setup_method(self):
        """
        Loads particle data into a SingleScatteringData object and
        stores it in the data attribute.
        """
        self.particle = utils.particle_random_1
        f_grid = self.particle.frequencies
        t_grid = self.particle.temperatures
        lon_inc = self.particle[0].lon_inc
        lat_inc = self.particle[0].lat_inc
        lon_scat = self.particle[0].lon_scat
        lat_scat = self.particle[0].lat_scat[:-1]

        self.f_grid = self.particle.frequencies
        self.t_grid = self.particle.temperatures
        self.data = SingleScatteringData(f_grid,
                                         t_grid,
                                         lon_inc,
                                         lat_inc,
                                         lon_scat,
                                         lat_scat,
                                         ParticleType.Random)
        for i, f in enumerate(self.particle.frequencies):
            for j, t in enumerate(self.particle.temperatures):
                data = self.particle.get_scattering_data(f, t)
                sd = particle_to_single_scattering_data_random(data, f, t)
                self.data.set_data(i, j, sd)

    def test_set_data(self):
        """
        Tests that interpolating the SingleScatteringData object to the
        frequencies and temperatures from the input data reproduces its
        values.
        """

        for f in self.f_grid:
            for t in self.t_grid:
                data = self.particle.get_scattering_data(f, t)
                sd_ref = particle_to_single_scattering_data_random(data, f, t)
                sd = self.data.interpolate_frequency([f]).interpolate_temperature([t])

                assert np.all(np.isclose(sd_ref.get_phase_matrix(),
                                         sd.get_phase_matrix()))
                assert np.all(np.isclose(sd_ref.get_extinction_matrix(),
                                         sd.get_extinction_matrix()))
                assert np.all(np.isclose(sd_ref.get_absorption_vector(),
                                         sd.get_absorption_vector()))
                assert np.all(np.isclose(sd_ref.get_forward_scattering_coeff(),
                                         sd.get_forward_scattering_coeff()))
                assert np.all(np.isclose(sd_ref.get_backward_scattering_coeff(),
                                         sd.get_backward_scattering_coeff()))

    def test_add_data(self):
        """
        Tests adding of data.
        """
        data_summed = self.data + self.data + self.data
        data_scaled = self.data * 3.0

        for f in self.f_grid:
            for t in self.t_grid:
                sd = data_summed.interpolate_frequency([f]).interpolate_temperature([t])
                sd_ref = data_scaled.interpolate_frequency([f]).interpolate_temperature([t])

                assert np.all(np.isclose(sd_ref.get_phase_matrix(),
                                         sd.get_phase_matrix()))
                assert np.all(np.isclose(sd_ref.get_extinction_matrix(),
                                         sd.get_extinction_matrix()))
                assert np.all(np.isclose(sd_ref.get_absorption_vector(),
                                         sd.get_absorption_vector()))
                assert np.all(np.isclose(sd_ref.get_forward_scattering_coeff(),
                                         sd.get_forward_scattering_coeff()))
                assert np.all(np.isclose(sd_ref.get_backward_scattering_coeff(),
                                         sd.get_backward_scattering_coeff()))

    def test_conversion(self):
        """
        Tests adding of data.
        """
        ssd_spectral = self.data.to_spectral()
        pm_gridded = self.data.get_phase_matrix()
        pm_spectral = ssd_spectral.get_phase_matrix()
        assert np.all(np.isclose(pm_gridded, pm_spectral))

        ssd_spectral_1 = self.data.to_spectral(32, 0)
        ssd_spectral_2 = ssd_spectral.to_spectral(32, 0)
        assert np.all(np.isclose(ssd_spectral_1.get_phase_matrix(),
                                 ssd_spectral_2.get_phase_matrix()))

def particle_to_single_scattering_data_azimuthally_random(particle_data,
                                                          frequency,
                                                          temperature):
    """
    Convert ssdb.Particle to SingleScatteringData object.
    """

    f_grid = np.array([frequency])
    t_grid = np.array([temperature])

    pm = particle_data.phase_matrix
    em = particle_data.extinction_matrix
    av = particle_data.absorption_vector
    bsc = particle_data.backward_scattering_coeff
    fsc = particle_data.forward_scattering_coeff

    n_coeffs = pm.shape[-2]
    l_max = SHT.calc_l_max(n_coeffs)
    n_lat = max(l_max + 2 + l_max % 2, 32)
    n_lon = 2 * l_max + 2
    sht = SHT(l_max, l_max, n_lat, n_lon)

    sd = SingleScatteringData(f_grid,
                              t_grid,
                              particle_data.lon_inc,
                              particle_data.lat_inc,
                              sht,
                              pm,
                              em,
                              av,
                              bsc,
                              fsc)
    return sd


class TestSingleScatteringDataAzimuthallyRandom:
    """
    Test loading and manipulation of azimuthally random single scattering data.

    Attributes:
        f_grid (np.array): The frequency grid on which the scattering data
            is defined.
        t_grid (np.array): The temperature grid on which the scattering data
            is defined.
        data (SingleScatteringData): Single scattering data object containing
            the test data.
        particle(ssdb.Particle): Particle object providing access to the test
            data.
    """

    def setup_method(self):
        """
        Loads particle data into a SingleScatteringData object and
        stores it in the data attribute.
        """
        self.particle = utils.particle_azimuthally_random_1
        f_grid = self.particle.frequencies
        t_grid = self.particle.temperatures
        lon_inc = self.particle[0].lon_inc
        lat_inc = self.particle[0].lat_inc

        n_coeffs = self.particle[0].phase_matrix.shape[-1]
        l_max = SHT.calc_l_max(n_coeffs)
        l_max = 32
        n_lat = l_max + 2 + l_max % 2
        n_lon = 2 * l_max + 2
        sht = SHT(l_max, l_max, n_lat, n_lon)

        self.f_grid = self.particle.frequencies
        self.t_grid = self.particle.temperatures
        self.data = SingleScatteringData(f_grid,
                                         t_grid,
                                         lon_inc,
                                         lat_inc,
                                         l_max,
                                         ParticleType.AzimuthallyRandom)
        for i, f in enumerate(self.particle.frequencies):
            for j, t in enumerate(self.particle.temperatures):
                data = self.particle.get_scattering_data(f, t)
                sd = particle_to_single_scattering_data_azimuthally_random(data, f, t)
                self.data.set_data(i, j, sd)

    def test_set_data(self):
        """
        Tests that interpolating the SingleScatteringData object to the
        frequencies and temperatures from the input data reproduces its
        values.
        """

        for f in self.f_grid:
            for t in self.t_grid:
                data = self.particle.get_scattering_data(f, t)
                sd_ref = particle_to_single_scattering_data_azimuthally_random(data,
                                                                               f,
                                                                               t)
                sd = self.data.interpolate_frequency([f]).interpolate_temperature([t])
                sd = sd.to_gridded()
                sd_ref = sd_ref.to_gridded()
                sd = sd.interpolate_angles(sd_ref.get_lon_inc(),
                                           sd_ref.get_lat_inc(),
                                           sd_ref.get_lon_scat(),
                                           sd_ref.get_lat_scat())

                return sd, sd_ref

                assert np.all(np.isclose(sd_ref.get_phase_matrix(),
                                         sd.get_phase_matrix()))
                assert np.all(np.isclose(sd_ref.get_extinction_matrix(),
                                         sd.get_extinction_matrix()))
                assert np.all(np.isclose(sd_ref.get_absorption_vector(),
                                         sd.get_absorption_vector()))
                assert np.all(np.isclose(sd_ref.get_forward_scattering_coeff(),
                                         sd.get_forward_scattering_coeff()))
                assert np.all(np.isclose(sd_ref.get_backward_scattering_coeff(),
                                         sd.get_backward_scattering_coeff()))

    def test_add_data(self):
        """
        Tests adding of data.
        """
        data_summed = self.data + self.data + self.data
        data_scaled = self.data * 3.0

        for f in self.f_grid:
            for t in self.t_grid:
                sd = data_summed.interpolate_frequency([f]).interpolate_temperature([t])
                sd_ref = data_scaled.interpolate_frequency([f]).interpolate_temperature([t])

                assert np.all(np.isclose(sd_ref.get_phase_matrix(),
                                         sd.get_phase_matrix()))
                assert np.all(np.isclose(sd_ref.get_extinction_matrix(),
                                         sd.get_extinction_matrix()))
                assert np.all(np.isclose(sd_ref.get_absorption_vector(),
                                         sd.get_absorption_vector()))
                assert np.all(np.isclose(sd_ref.get_forward_scattering_coeff(),
                                         sd.get_forward_scattering_coeff()))
                assert np.all(np.isclose(sd_ref.get_backward_scattering_coeff(),
                                         sd.get_backward_scattering_coeff()))

    def test_conversion(self):
        """
        Tests adding of data.
        """
        ssd_spectral = self.data.to_spectral()
        pm_gridded = self.data.get_phase_matrix()
        pm_spectral = ssd_spectral.get_phase_matrix()

        assert np.all(np.isclose(pm_gridded, pm_spectral))

        ssd_spectral_1 = self.data.to_spectral(32, 0)
        ssd_spectral_2 = ssd_spectral.to_spectral(32, 0)
        assert np.all(np.isclose(ssd_spectral_1.get_phase_matrix(),
                                 ssd_spectral_2.get_phase_matrix()))
