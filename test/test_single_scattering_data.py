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
                                            ParticleType)

#Import test utils.
try:
    sys.path.append(os.path.dirname(__file__))
except Exception:
    pass

import utils

def particle_to_single_scattering_data(particle_data,
                                       frequency,
                                       temperature):
    """
    Convert ssdb.Particle to SingleScatteringData object.
    """
    f_grid = np.array([frequency])
    t_grid = np.array([temperature])

    pm = np.transpose(particle_data.phase_matrix.data, [1, 2, 3, 4, 0])
    pm = pm.reshape((1, 1) + pm.shape)
    pm = pm[..., :1]

    em = np.transpose(particle_data.extinction_matrix.data, [1, 2, 0])
    em = em.reshape((1, 1) + em.shape + (1, 1))
    em = em[..., :1]

    av = np.transpose(particle_data.absorption_vector.data, [1, 2, 0])
    av = av.reshape((1, 1) + av.shape + (1, 1))
    av = av[..., :1]

    bsc = pm[:, :, :, :, 0, -1, 0]
    fsc = pm[:, :, :, :, 0, 0, 0]

    sd = SingleScatteringData(f_grid,
                              t_grid,
                              particle_data.lon_inc,
                              particle_data.lat_inc,
                              particle_data.lon_scat,
                              particle_data.lat_scat,
                              pm,
                              em,
                              av,
                              bsc,
                              fsc)
    return sd

class TestSingleScatteringDataGridded:

    def test_set_data(self):
        """
        Tests setting of data for given frequency and temperature indices.
        """
        p = utils.particle_spherical_1
        f_grid = p.frequencies
        t_grid = p.temperatures
        lon_inc = p[0].lon_inc
        lat_inc = p[0].lat_inc
        lon_scat = p[0].lon_scat
        lat_scat = p[0].lat_scat

        ssd = SingleScatteringData(f_grid,
                                   t_grid,
                                   lon_inc,
                                   lat_inc,
                                   lon_scat,
                                   lat_scat,
                                   ParticleType.Spherical)
        for i, f in enumerate(p.frequencies):
            for j, t in enumerate(p.temperatures):
                data = p.get_scattering_data(f, t)
                sd = particle_to_single_scattering_data(data, f, t)
                ssd.set_data(i, j, sd)

        for i, f in enumerate(p.frequencies):
            for j, t in enumerate(p.temperatures):
                data = p.get_scattering_data(f, t)
                sd_ref = particle_to_single_scattering_data(data, f, t)
                sd = ssd.interpolate_frequency([f]).interpolate_temperature([t])

                assert np.all(np.isclose(sd_ref.get_phase_matrix(),
                                         sd.get_phase_matrix()))
                assert np.all(np.isclose(sd_ref.get_extinction_matrix(),
                                         sd.get_extinction_matrix()))
                assert np.all(np.isclose(sd_ref.get_absorption_vector(),
                                         sd.get_absorption_vector()))
