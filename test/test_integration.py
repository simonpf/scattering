import numpy as np
import scipy as sp
from scipy.special import roots_legendre

from scatlib.integration import RegularQuadrature, GaussLegendreQuadrature

def test_regular_quadrature():
    n = np.random.randint(10, 100)
    q = RegularQuadrature(n)
    nodes = q.get_nodes()
    weights = q.get_weights()
    nodes_ref, weights_ref = roots_legendre(n)

    return nodes, nodes_ref

def test_gauss_legendre_quadrature():
    n = np.random.randint(10, 100)
    q = GaussLegendreQuadrature(n)
    nodes = q.get_nodes()
    weights = q.get_weights()
    nodes_ref, weights_ref = roots_legendre(n)

    return nodes, nodes_ref
