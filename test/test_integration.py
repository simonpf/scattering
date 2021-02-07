import numpy as np
import scipy as sp
from scipy.special import roots_legendre, lpmn

from scattering.integration import (ClenshawCurtisQuadrature,
                                    FejerQuadrature,
                                    GaussLegendreQuadrature,
                                    DoubleGaussQuadrature,
                                    LobattoQuadrature,
                                    GaussLegendreLatitudeGrid,
                                    DoubleGaussLatitudeGrid,
                                    LobattoLatitudeGrid)

def test_clenshaw_curtis_quadrature():
    """
    Ensure that the Clenshaw-Curtis quadrature return the expected
    nodes and the integrating sin(np.arcos(x)) over -1, 1 yields th
    expected result.
    """
    n = np.random.randint(10, 100)
    q = ClenshawCurtisQuadrature(n)
    nodes = q.get_nodes()
    weights = q.get_weights()

    nodes_ref = -np.cos(np.pi * np.arange(0, n) / (n - 1))

    assert np.all(np.isclose(nodes, nodes_ref))

    fx = np.sin(np.arccos(nodes_ref))
    integral = np.sum(fx * weights)
    assert np.isclose(integral, 0.5 * np.pi)

def test_fejer_quadrature():
    """
    Ensure that the Fejer quadrature returns the expected nodes
    and that integrating sin(np.arcos(x)) over -1, 1 yields the
    expected results.
    """
    n = np.random.randint(10, 100)
    q = FejerQuadrature(n)
    nodes = q.get_nodes()
    weights = q.get_weights()
    nodes_ref = -np.cos(np.pi * (np.arange(0, n) + 0.5) / n)
    assert np.all(np.isclose(nodes, nodes_ref))

    fx = np.sin(np.arccos(nodes_ref))
    integral = np.sum(fx * weights)
    assert np.isclose(integral, 0.5 * np.pi)

def test_gauss_legendre_quadrature():
    """
    Ensure that Gauss-Legendre quadrature nodes and weights are
    the same as when computed using scipy.
    """
    n = np.random.randint(10, 100)
    q = GaussLegendreQuadrature(n)
    nodes = q.get_nodes()
    weights = q.get_weights()
    nodes_ref, weights_ref = roots_legendre(n)

    assert np.all(np.isclose(nodes, nodes_ref))
    assert np.all(np.isclose(weights, weights_ref))


def test_double_gauss_quadrature():
    """
    Ensure that nodes and weights of the double Gauss quadrature
    match those of a scaled and shifted Gauss-Legendre quadrature.
    """
    n = 2 * np.random.randint(10, 100)
    q = DoubleGaussQuadrature(n)
    nodes = q.get_nodes()
    weights = q.get_weights()
    nodes_ref, weights_ref = roots_legendre(n // 2)

    assert np.all(np.isclose(-0.5 + 0.5 * nodes_ref,
                             nodes[: n // 2]))
    assert np.all(np.isclose(0.5 * weights_ref,
                             weights[: n // 2]))
    assert np.all(np.isclose(0.5 + 0.5 * nodes_ref,
                             nodes[n // 2:]))
    assert np.all(np.isclose(0.5 * weights_ref,
                             weights[n // 2:]))

def test_lobatto_quadrature():
    """
    Ensure that values of the Lobatto quadrature match the value expected
    for a specific degree.

    Reference values taken from: https://mathworld.wolfram.com/LobattoQuadrature.html
    """
    n = 5
    q = LobattoQuadrature(n)

    nodes = q.get_nodes()
    weights = q.get_weights()

    nodes_ref = np.array([-1.0, -np.sqrt(21) / 7, 0, np.sqrt(21) / 7, 1])
    weights_ref = np.array([0.1, 49 / 90, 32 / 45, 49 / 90, 0.1])

    print(weights, weights_ref)
    assert np.all(np.isclose(nodes, nodes_ref))
    assert np.all(np.isclose(weights, weights_ref))

    latitude_grid = LobattoLatitudeGrid(5)
    colats = latitude_grid.get_colatitude_grid()
    lats = latitude_grid.get_latitude_grid()

    assert np.all(np.isclose(colats, nodes))


