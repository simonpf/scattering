import numpy as np
import matplotlib.pyplot as plt
import pytest
import scipy
from scatlib.sht import SHT
from scipy.special import roots_legendre, sph_harm

class TestSHT:

    def setup_method(self):
        self.l_max = 2 ** np.random.randint(4, 8)
        self.m_max = self.l_max
        self.n_lat = 2 * self.l_max
        self.n_lon = 4 * self.l_max
        self.sht = SHT(self.l_max, self.m_max, self.n_lat, self.n_lon, 1)

    def test_spatial_to_spectral(self):
        l = np.random.randint(1, self.l_max)
        m = np.random.randint(-l, l)

        lat_grid, _ = np.arccos(roots_legendre(self.n_lat))
        lon_grid = np.linspace(0, 2.0 * np.pi, self.n_lon + 1)[:-1]
        xx, yy = np.meshgrid(lat_grid, lon_grid, indexing="xy")
        zz = sph_harm(m, l, yy, xx) 
        coeffs = self.sht.transform(zz.real)

        assert(np.sum(np.abs(coeffs > 1e-6)) == 1)

    def test_inverse_transform(self):
        l = np.random.randint(1, self.l_max)
        m = np.random.randint(-l, l)
        lat_grid, _ = np.arccos(roots_legendre(self.n_lat))
        lon_grid = np.linspace(0, 2.0 * np.pi, self.n_lon + 1)[:-1]

        xx, yy = np.meshgrid(lon_grid, lat_grid, indexing="ij")
        zz = sph_harm(m, l, xx, yy)
        coeffs = self.sht.transform(self.sht.transform(zz.real))
        assert(np.all(np.isclose(zz.real, coeffs)))
