import numpy as np
import scipy as sp
from scipy.special import roots_legendre

from scatlib.itegration import GaussLegendreQuadrature

def test_nodes_and_weights():
    n = np.random.randint(10, 100)
    q = GaussLegendreQuadrature(n)
    nodes = q.get_nodes()
    weights = q.get_weights()
    nodes_ref, weights_ref, mu = roots_legendre(n)
    return nodes, nodes_ref

