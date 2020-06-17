Spherical harmonics transform
#############################

A key-feature of scatlib is that it provides seamless handling of gridded and
spectral scattering data. For efficient conversions between the two formats
scatlib is built ontop of the `SHTns <https://nschaeff.bitbucket.io/shtns/>`_
library.

Basics
------

Spherical harmonic functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The spherical harmonics function of *degree* :math:`l` and order :math:`m` is defined as

.. math::

  Y_l^m(\theta, \phi) = N\ \exp(i m \phi) P_l^m(\cos(\theta))

where :math:`N` is a normalization constant and :math:`P_l^m` an associated
Legendre polynomial

.. math::

  P^m_l(x) = (-1)^m(1 - x^2)^{\frac{m}{2}}\frac{d^m}{dx^m}(P_l(x))

with :math:`P_l` an ordinary Legendre polynomial.

Spherical harmonics expansion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any square-integrable function can be expanded as a linear combination of spherical harmonics:

.. math::

  f(\theta, \phi) = \sum_{l = 0}^\infty \sum_{m = -l}^{l} f_l^m Y_l^m(\theta, \phi)

The expansion coefficients :math:`f_l^m` can be obtained by computing the spherical harmonics
transform

.. math::

   f_l^m = \int_0^{2\pi}\ d\phi \int_0^\pi\ d\theta \ f(\theta, \phi) Y_l^m(\theta, \phi)^*

Discretization
~~~~~~~~~~~~~~

The spherical harmonics expansion can be used to represent data on a spherical grid. For
numerical reasons it is assumed that the grid is regular, i.e. that it can
be described by two vectors.


The colatitude vector :math:`\boldsymbol{\theta} = [\theta_1, \ldots, \theta_n]`
and the longitude vector :math:`\boldsymbol{\phi} = [\phi_1, \ldots, \phi_m] `.
Furthermore, the longitude grid must be homogeneous (fixed spacing) and the
colatitude grid should adhere to the assumptions of numerical integration scheme
used to compute the expansion.

Anti-aliasing conditions
~~~~~~~~~~~~~~~~~~~~~~~~

In order to compute a spherical harmonics expansion with maximum-degree
:math:`l_\text{max]` and maximum order :math:`m_\text{max}`, the resolutions of
the spatial grid, given by vectors :math:`\boldsymbol{\theta}` and
:math`\boldsymbol{\phi}`, must satisfy the following anti-aliasing conditions:

.. math::

  \text{len}(\theta) \geq l_\text{max} \\
  \text{len}(\phi) \geq 2 m_\text{max}

   
   

