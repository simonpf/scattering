"""
Tests the ScatteringData C++ class using azimuthally random scattering data.
"""
import sys
import os
import netCDF4
import numpy as np
import scipy as sp
import scipy.interpolate
from scatlib.scattering_data import ScatteringDataGridded, gridded_to_spectral

# Import test utils.
try:
    sys.path.append(os.path.dirname(__file__))
except Exception:
    pass

import utils

def particle_to_single_scattering_data(frequency, temperature, ):

    f_grid = np.array([frequency])
    t_grid = np.array([temperature])

    pm = np.transpose(p.phase_matrix.data, [4, 1, 2, 3, 0])
    pm = pm.reshape((1, 1) + pm.shape)

    em = np.transpose(p.extinction_matrix.data, [2, 1, 2, 0])
    em = em.reshape((1, 1) + em.shape + (1, 1))

    av = np.transpose(p.absorption_vector.data, [2, 1, 2, 0])
    av = av.reshape((1, 1) + em.shape + (1, 1))

    bsc = pm[:, :, :, :, 0, -1, 0]
    fsc = pm[:, :, :, :, 0, 0, 0]

    sd = SingleScatteringData(f_grid,
                              t_grid,
                              p.lon_inc,
                              p.lat_inc,
                              p.lon_scat,
                              p.lat_scat,
                              pm,
                              em,
                              av,
                              bsc,
                              fsc)
    return sd

class TestSingleScatteringDataGridded:

    def test_set_data(self):
        p = utils.particle_spherical_1
        f_grid = p.frequencies
        t_grid = p.temperatures
        lon_inc = p[0].lon_inc
        lat_inc = p[0].lat_inc
        lon_scat = p[0].lon_scat
        lat_scat = p[0].lat_scat
        n_elements = p.scattering_data.shape[0]

        ssd = SingleScatteringData(f_grid,
                                   t_grid,
                                   lon_inc,
                                   lat_inc,
                                   lon_scat,
                                   lat_scat,
                                   1)
        for i, f in enumerate(p.frequencies):
            for j, t in enumerate(p.temperatures):
                sd = particle_to_single_scattering_data(f, t, p[i, j])
                ssd.set_data(i, j) = sd
