"""
Test interface to ARTS single scattering database (SSDB).
"""
import scattering.arts_ssdb as ssdb
import os
import utils
from utils import RANDOM_DATA_PATH, AZIMUTHALLY_RANDOM_DATA_PATH
import numpy as np

def test_load_random_particle():
    """
    Test loading of random particle data.
    """
    path = os.path.join(RANDOM_DATA_PATH,
                        "Dmax00688um_Dveq00361um_Mass2.25360e-08kg.nc")
    particle_file = ssdb.ParticleFile(path)
    temps = particle_file.get_temperatures()
    freqs = particle_file.get_frequencies()
    t_grid = particle_file.get_t_grid()
    f_grid = particle_file.get_f_grid()

    assert np.all(np.isclose(temps, t_grid))
    assert np.all(np.isclose(freqs, f_grid))

    particle_data = particle_file.to_single_scattering_data()
    lon_inc_particle = particle_data.get_lon_inc()
    lat_inc_particle = particle_data.get_lat_inc()
    lon_scat_particle = particle_data.get_lon_scat()
    lat_scat_particle = particle_data.get_lat_scat()

    for i, f in enumerate(freqs):
        for j, t in enumerate(temps):
            data = particle_file.get_scattering_data(i, j)
            lon_inc = data.get_lon_inc()
            lat_inc = data.get_lat_inc()
            lon_scat = data.get_lon_inc()
            lat_scat = data.get_lat_scat()

            assert lon_inc.size == 1
            assert lat_inc.size == 1
            assert lon_scat.size == 1

            assert lon_inc_particle.size >= lon_inc.size
            assert lat_inc_particle.size >= lat_inc.size
            assert lon_scat_particle.size >= lon_scat.size
            assert lat_scat_particle.size >= lat_scat.size

            data.get_phase_matrix_data_gridded()
            data.get_extinction_matrix_data_gridded()
            data.get_absorption_vector_data_gridded()
            data.get_backward_scattering_coeff_data_gridded()
            data.get_forward_scattering_coeff_data_gridded()

            assert f * 1e9 == data.get_frequency()
            assert t == data.get_temperature()

            assert data.get_particle_type() == ssdb.ParticleType.Random


def test_load_azimuthally_random_particle():
    """
    Test loading of azimuthally random particle data.
    """
    path = os.path.join(AZIMUTHALLY_RANDOM_DATA_PATH,
                        "Dmax00590um_Dveq00251um_Mass7.59425e-09kg.nc")
    particle_file = ssdb.ParticleFile(path)
    temps = particle_file.get_temperatures()
    freqs = particle_file.get_frequencies()
    t_grid = particle_file.get_t_grid()
    f_grid = particle_file.get_f_grid()

    particle_data = particle_file.to_single_scattering_data()
    lon_inc_particle = particle_data.get_lon_inc()
    lat_inc_particle = particle_data.get_lat_inc()
    lon_scat_particle = particle_data.get_lon_scat()
    lat_scat_particle = particle_data.get_lat_scat()

    assert np.all(np.isclose(temps, t_grid))
    assert np.all(np.isclose(freqs, f_grid))

    for i, f in enumerate(freqs):
        for j, t in enumerate(temps):
            data = particle_file.get_scattering_data(i, j)
            lon_inc = data.get_lon_inc()
            lat_inc = data.get_lat_inc()

            assert lon_inc.size == 1

            assert lon_inc_particle.size >= lon_inc.size
            assert lat_inc_particle.size >= lat_inc.size

            data.get_phase_matrix_data_spectral()
            data.get_extinction_matrix_data_spectral()
            data.get_absorption_vector_data_spectral()
            data.get_backward_scattering_coeff_data_spectral()
            data.get_forward_scattering_coeff_data_spectral()

            assert f * 1e9 == data.get_frequency()
            assert t == data.get_temperature()

            assert data.get_particle_type() == ssdb.ParticleType.AzimuthallyRandom
