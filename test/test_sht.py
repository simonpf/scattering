"""
Test for the SHTns C++ interface.
"""
import numpy as np
import matplotlib.pyplot as plt
import pytest
import scipy
from scatlib.sht import SHT
from scipy.special import roots_legendre, sph_harm

class TestSHT:
    """
    Testing of spherical harmonics transform for non-degenrate 2D fields.
    """
    def setup_method(self):
        self.l_max = np.random.randint(20, 100)
        self.m_max = np.random.randint(10, self.l_max)
        self.m_max = self.l_max #np.random.randint(10, self.l_max)

        self.n_lat = 2 * self.l_max
        self.n_lon = 4 * self.l_max
        self.sht = SHT(self.l_max, self.m_max, self.n_lon, self.n_lat)
        self.lat_grid, _ = np.sort(np.arccos(roots_legendre(self.n_lat)))
        self.lon_grid = np.linspace(0, 2.0 * np.pi, self.n_lon + 1)[:-1]

    def test_latitude_grid(self):
        """
        Test that the SHT class returns the expected latitude grid (Gauss-Legendre).
        """
        lat_grid_ref = self.lat_grid
        lat_grid = self.sht.get_latitude_grid()
        assert np.all(np.isclose(lat_grid_ref, lat_grid))

    def test_colatitude_grid(self):
        """
        Test that the SHT class returns the expected co-latitude grid (Gauss-Legendre).
        """
        lat_grid_ref = np.cos(self.lat_grid)
        lat_grid = self.sht.get_colatitude_grid()
        assert np.all(np.isclose(lat_grid_ref, lat_grid))

    def test_spatial_to_spectral(self):
        """
        Test transform from spatial to spectral representation by transforming
        spherical harmonics and ensuring that the result contains only one
        significant component.
        """
        l = np.random.randint(1, self.l_max)
        m_max = min(self.m_max, l)
        m_max = l
        m = np.random.randint(-m_max, m_max)

        xx, yy = np.meshgrid(self.lat_grid, self.lon_grid, indexing="xy")
        zz = sph_harm(m, l, yy, xx)
        coeffs = self.sht.transform(zz.real)

        assert np.sum(np.abs(coeffs) > 1e-6) == 1

    def test_inverse_transform(self):
        """
        Test transforming from spatial field to spectral field and back to ensure
        that the input field is recovered.
        """
        l = np.random.randint(1, self.l_max)
        m_max = min(self.m_max, l)
        m_max = l
        m = np.random.randint(-m_max, m_max)

        xx, yy = np.meshgrid(self.lon_grid, self.lat_grid, indexing="ij")
        zz = sph_harm(m, l, xx, yy)
        coeffs = self.sht.synthesize(self.sht.transform(zz.real).ravel())
        assert np.all(np.isclose(zz.real, coeffs))

    def test_inverse_transform_cmplx(self):
        """
        Test transforming from spatial field to spectral field and back to ensure
        that the input field is recovered.
        """
        l = np.random.randint(1, self.l_max)
        m_max = l#min(self.m_max, l)
        m = np.random.randint(-m_max, m_max)

        xx, yy = np.meshgrid(self.lon_grid, self.lat_grid, indexing="ij")
        zz = sph_harm(m, l, xx, yy)
        coeffs = self.sht.transform_cmplx(zz)
        zz_rec = self.sht.synthesize_cmplx(2.0 * coeffs)
        assert np.all(np.isclose(2.0 * zz, zz_rec))

    def test_evaluate(self):
        """
        Test that evaluating the spectral representation reproduces the spatial
        input.
        """
        l = np.random.randint(1, self.l_max)
        m_max = min(self.m_max, l)
        m_max = l
        m = np.random.randint(-m_max, m_max)

        xx, yy = np.meshgrid(self.lat_grid, self.lon_grid, indexing="xy")
        zz_ref = sph_harm(m, l, yy, xx) 
        coeffs = self.sht.transform(zz_ref.real)
        points = np.stack([yy.ravel(), xx.ravel()], axis=-1)
        zz = self.sht.evaluate(coeffs, points)

        assert np.all(np.isclose(zz, zz_ref.real.ravel()))

class TestLegendreExpansion:
    """
    Testing of spherical harmonics transform for 1D fields.
    """
    def setup_method(self):
        self.l_max = 2 ** np.random.randint(4, 8)
        self.m_max = 0
        self.n_lat = 2 * self.l_max
        self.n_lon = 1
        self.sht = SHT(self.l_max, self.m_max, self.n_lon, self.n_lat)
        self.lat_grid, _ = np.sort(np.arccos(roots_legendre(self.n_lat)))
        self.lon_grid = np.linspace(0, 2.0 * np.pi, self.n_lon + 1)[:-1]

    def test_legendre_transform(self):
        """
        Tests the Legendre expansion by transforming a given Legendre polynomial
        and ensuring that only one component in the output is set.
        """
        l = np.random.randint(1, self.l_max)
        m = 0

        xx, yy = np.meshgrid(self.lat_grid, self.lon_grid, indexing="xy")
        zz = sph_harm(m, l, yy, xx)
        coeffs = self.sht.transform(zz.real)

        assert np.sum(np.abs(coeffs) > 1e-6) == 1

    def test_legendre_transform_cmplx(self):
        """
        Test transforming from complex spatial field to spectral field and back to ensure
        that the input field is recovered.
        """
        l = np.random.randint(1, self.l_max)
        m = 0

        xx, yy = np.meshgrid(self.lat_grid, self.lon_grid, indexing="xy")
        zz = sph_harm(m, l, yy, xx)
        aa = self.sht.transform_cmplx(zz)
        zz_rec = self.sht.synthesize_cmplx(2.0 * self.sht.transform_cmplx(zz))

        assert np.all(np.isclose(2.0 * zz, zz_rec))


    def test_evaluate(self):
        """
        Test that evaluating the spectral representation reproduces the spatial
        input.
        """
        l = np.random.randint(1, self.l_max)
        m = 0

        xx, yy = np.meshgrid(self.lat_grid, self.lon_grid, indexing="xy")
        zz_ref = sph_harm(m, l, yy, xx) 
        coeffs = self.sht.transform(zz_ref.real)
        points = xx.ravel()
        zz = self.sht.evaluate(coeffs, points)

        assert np.all(np.isclose(zz, zz_ref.real.ravel()))

class TestTrivalTransform:
    """
    Testing of spherical harmonics transform for 1D fields.
    """
    def setup_method(self):
        self.l_max = 0
        self.m_max = 0
        self.n_lat = 1
        self.n_lon = 1
        self.sht = SHT(self.l_max, self.m_max, self.n_lon, self.n_lat)

    def test_transform(self):
        """
        Tests the Legendre expansion by transforming a given Legendre polynomial
        and ensuring that only one component in the output is set.
        """
        z = np.ones((1, 1))
        zz = self.sht.synthesize(self.sht.transform(z))

        assert np.all(np.isclose(z, zz))

    def test_evaluate(self):
        """
        Test that evaluating the spectral representation reproduces the spatial
        input.
        """
        coeffs = np.ones(1)
        points = np.random.rand(10, 2)
        zz = self.sht.evaluate(coeffs, points)

        assert np.all(np.isclose(zz, 1.0))

def synthesize_matrix(sht_inc, sht_scat, coefficient_matrix):
    n = coefficient_matrix.shape[1]
    coeffs = np.zeros((sht_inc.get_n_longitudes(),
                      sht_inc.get_n_latitudes(),
                      n)) + 0j
    for i in range(n):
        coeffs[:, :, i] = sht_inc.synthesize_cmplx(coefficient_matrix[:, i])
    result = np.zeros((sht_inc.get_n_longitudes(),
                       sht_inc.get_n_latitudes(),
                       sht_scat.get_n_longitudes(),
                       sht_scat.get_n_latitudes()))
    for i in range(coeffs.shape[0]):
        for j in range(coeffs.shape[1]):
            result[i, j] = sht_scat.synthesize(coeffs[i, j, :])
    return result

def transform_matrix(sht_inc, sht_scat, spatial_field):
    coeffs = np.zeros((sht_inc.get_n_longitudes(),
                       sht_inc.get_n_latitudes(),
                       sht_scat.get_n_spectral_coeffs())) + 0j
    for i in range(sht_inc.get_n_longitudes()):
        for j in range(sht_inc.get_n_latitudes()):
            coeffs[i, j, :] = sht_scat.transform(spatial_field[i, j, :, :])

    result = np.zeros((sht_inc.get_n_spectral_coeffs_cmplx(),
                       sht_scat.get_n_spectral_coeffs()))
    result = result + 0j

    for i in range(sht_scat.get_n_spectral_coeffs()):
        result[:, i] = sht_inc.transform_cmplx(coeffs[:, :, i])

    return result


class TestAddition:
    """
    Test addition of SHT coefficient vectors and matrices.
    """
    def setup_method(self):
        n_lon = 64
        n_lat = 64
        self.l_max_l = np.random.randint(10, 20)
        self.m_max_l = self.l_max_l#np.random.randint(9, self.l_max_l)
        self.sht_l = SHT(self.l_max_l, self.m_max_l, n_lon, n_lat)

        self.l_max_l_inc = np.random.randint(5, 10)
        self.m_max_l_inc = self.l_max_l_inc #np.random.randint(4, self.l_max_l_inc)
        self.sht_l_inc = SHT(self.l_max_l_inc, self.m_max_l_inc, n_lon, n_lat)

        self.l_max_r = np.random.randint(20, 24)
        self.m_max_r = self.l_max_r #np.random.randint(19, self.l_max_r)
        self.sht_r = SHT(self.l_max_r, self.m_max_r, n_lon, n_lat)

        self.l_max_r_inc = np.random.randint(10, 20)
        self.m_max_r_inc = self.l_max_r_inc #np.random.randint(9, self.l_max_r_inc)
        self.sht_r_inc = SHT(self.l_max_r_inc, self.m_max_r_inc, n_lon, n_lat)

        self.coeff_vector_l = np.random.rand(self.sht_l.get_n_spectral_coeffs())
        self.coeff_vector_l = self.coeff_vector_l + 1j * np.random.rand(self.sht_l.get_n_spectral_coeffs())
        self.coeff_matrix_l = np.random.rand(self.sht_l_inc.get_n_spectral_coeffs_cmplx(),
                                             self.sht_l.get_n_spectral_coeffs())
        self.coeff_matrix_l = self.coeff_matrix_l + 1j * np.random.rand(self.sht_l_inc.get_n_spectral_coeffs_cmplx(),
                                                                        self.sht_l.get_n_spectral_coeffs())

        self.coeff_vector_r = np.random.rand(self.sht_r.get_n_spectral_coeffs())
        self.coeff_vector_r = self.coeff_vector_r + 1j * np.random.rand(self.sht_r.get_n_spectral_coeffs())
        self.coeff_matrix_r = np.random.rand(self.sht_r_inc.get_n_spectral_coeffs_cmplx(),
                                             self.sht_r.get_n_spectral_coeffs())
        self.coeff_matrix_r = self.coeff_matrix_r + 1j * np.random.rand(self.sht_r_inc.get_n_spectral_coeffs_cmplx(),
                                                                        self.sht_r.get_n_spectral_coeffs())

    def test_addition(self):
        """
        Test addition of coefficient vectors.
        """
        sum_2l = SHT.add_coeffs(self.sht_l, self.coeff_vector_l, self.sht_l, self.coeff_vector_l)
        sum_r = SHT.add_coeffs(self.sht_r, self.coeff_vector_r, self.sht_l, self.coeff_vector_l)
        sum_2r = SHT.add_coeffs(self.sht_r, self.coeff_vector_r, self.sht_r, self.coeff_vector_r)


        z_l = self.sht_l.synthesize(self.coeff_vector_l)
        z_r = self.sht_r.synthesize(self.coeff_vector_r)
        z_2l = self.sht_l.synthesize(sum_2l)
        z_2r = self.sht_r.synthesize(sum_2r)

        z_rs = self.sht_r.synthesize(sum_r)

        coeff_vector_rl = self.sht_l.transform(z_r)
        z_rl = self.sht_l.synthesize(coeff_vector_rl)
        sum_l = SHT.add_coeffs(self.sht_l, self.coeff_vector_l, self.sht_l, coeff_vector_rl)
        z_ls = self.sht_l.synthesize(sum_l)

        assert np.all(np.isclose(2.0 * z_l, z_2l))
        assert np.all(np.isclose(2.0 * z_r, z_2r))
        assert np.all(np.isclose(z_rs, z_r + z_l))
        assert np.all(np.isclose(z_ls, z_l + z_rl))


    def test_addition_matrix(self):
        """
        Test addition of coefficient vectors.
        """
        sum_2l = SHT.add_coeffs(self.sht_l_inc,
                                self.sht_l,
                                self.coeff_matrix_l,
                                self.sht_l_inc,
                                self.sht_l,
                                self.coeff_matrix_l)

        z_l = synthesize_matrix(self.sht_l_inc, self.sht_l, self.coeff_matrix_l)
        z_2l = synthesize_matrix(self.sht_l_inc, self.sht_l, sum_2l)
        assert np.all(np.isclose(2.0 * z_l, z_2l))

        z_r = synthesize_matrix(self.sht_r_inc, self.sht_r, self.coeff_matrix_r)
        sum_rl = SHT.add_coeffs(self.sht_r_inc,
                                self.sht_r,
                                self.coeff_matrix_r,
                                self.sht_l_inc,
                                self.sht_l,
                                self.coeff_matrix_l)
        z_rs = synthesize_matrix(self.sht_r_inc, self.sht_r, sum_rl)
        assert np.all(np.isclose(z_r + z_l, z_rs))

        coeff_matrix_rl = transform_matrix(self.sht_l_inc, self.sht_l, z_r)
        sum_lr = SHT.add_coeffs(self.sht_l_inc,
                                self.sht_l,
                                self.coeff_matrix_l,
                                self.sht_l_inc,
                                self.sht_l,
                                coeff_matrix_rl)
        z_ls = synthesize_matrix(self.sht_l_inc, self.sht_l, sum_lr)
        z_rl = synthesize_matrix(self.sht_l_inc, self.sht_l, coeff_matrix_rl)

        assert np.all(np.isclose(z_ls, z_rl + z_l))
