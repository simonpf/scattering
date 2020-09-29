/** \file scatlib/integration.h
 *
 * Quadratures and integration functions.
 *
 * @author Simon Pfreundschuh, 2020
 */
#include <map>

#include "eigen.h"

#ifndef __SCATLIB_INTEGRATION__
#define __SCATLIB_INTEGRATION__

namespace scatlib {
namespace detail {

////////////////////////////////////////////////////////////////////////////////
/// Type trait storing the desired precision for different precisions.
////////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
struct Precision {
  static constexpr Scalar value = 1e-16;
};

template <>
struct Precision<float> {
  static constexpr float value = 1e-6;
};

template <>
struct Precision<long double> {
  static constexpr long double value = 1e-19;
};
}  // namespace detail

// pxx :: export
// pxx :: instance(["double"])
/** Gauss-Legendre Quadarature
 *
 * This class implements a Gauss-Legendre for the integration of
 * functions of the interval [-1, 1].
 */
template <typename Scalar>
class GaussLegendreQuadrature {
 private:
  /** Find Gauss-Legendre nodes and weights.
   *
   * Uses the Newton root finding algorithm to find the roots of the
   * Legendre polynomial of degree n. Legendre functions are evaluated
   * using a recursion relation.
   */
  // pxx :: hide
  void calculate_nodes_and_weights() {
    const long int n = degree_;
    const long int n_half_nodes = (n + 1) / 2;
    const long int n_max_iter = 10;
    long int l, m, k;
    Scalar x, x_old, p_l, p_l_1, p_l_2, dp_dx;
    Scalar precision = detail::Precision<Scalar>::value;

    for (int i = 1; i <= n_half_nodes; ++i) {
      p_l = M_PI;
      p_l_1 = 2 * n;
      //
      // Initial guess.
      //
      x = -(1.0 - (n - 1) / (p_l_1 * p_l_1 * p_l_1)) *
          cos((p_l * (4 * i - 1)) / (4 * n + 2));

      //
      // Evaluate Legendre Polynomial and its derivative at node.
      //
      for (int j = 0; j < n_max_iter; ++j) {
        p_l = x;
        p_l_1 = 1.0;
        for (int l = 2; l <= n; ++l) {
          // Legendre recurrence relation
          p_l_2 = p_l_1;
          p_l_1 = p_l;
          p_l = ((2.0 * l - 1.0) * x * p_l_1 - (l - 1.0) * p_l_2) / l;
        }
        dp_dx = ((1.0 - x) * (1.0 + x)) / (n * (p_l_1 - x * p_l));
        x_old = x;

        //
        // Perform Newton step.
        //
        x -= p_l * dp_dx;
        auto dx = std::abs(x - x_old);
        auto threshold = 0.5 * (x + x_old);
        if (dx < threshold) {
          break;
        }
      }
      nodes_[i - 1] = x;
      weights_[i - 1] = 2.0 * dp_dx * dp_dx / ((1.0 - x) * (1.0 + x));
      nodes_[n - i] = -x;
      weights_[n - i] = weights_[i - 1];
    }
  }

 public:
  GaussLegendreQuadrature() {}
  GaussLegendreQuadrature(int degree)
      : degree_(degree), nodes_(degree), weights_(degree) {
    calculate_nodes_and_weights();
  }

  const eigen::Vector<Scalar>& get_nodes() const { return nodes_; }
  const eigen::Vector<Scalar>& get_weights() const { return weights_; }

  int degree_;
  eigen::Vector<Scalar> nodes_;
  eigen::Vector<Scalar> weights_;
};

template <typename Scalar>
class QuadratureProvider {
 public:
  QuadratureProvider() {}

  GaussLegendreQuadrature<Scalar> get_quadrature(int degree) {
    auto found = quadratures_.find(degree);
    if (found != quadratures_.end()) {
      return found->second;
    } else {
      quadratures_.insert({degree, GaussLegendreQuadrature<Scalar>(degree)});
      return quadratures_[degree];
    }
  }

 private:
  std::map<int, GaussLegendreQuadrature<Scalar>> quadratures_;
};

////////////////////////////////////////////////////////////////////////////////
// Integration functions
////////////////////////////////////////////////////////////////////////////////

static QuadratureProvider<double> quadratures = QuadratureProvider<double>();

template <typename Scalar>
Scalar integrate_latitudes(eigen::ConstVectorRef<Scalar> data,
                           const GaussLegendreQuadrature<Scalar> & quadrature) {
  auto weights = quadrature.get_weights();
  return weights.dot(data);
}

template <typename Scalar>
Scalar integrate_angles(eigen::ConstMatrixRef<Scalar> data,
                        eigen::ConstVectorRef<Scalar> longitudes,
                        eigen::ConstVectorRef<Scalar> colatitudes) {
  Scalar result = 0.0;
  eigen::Index n = longitudes.size();
  auto quadrature = quadratures.get_quadrature(colatitudes.size());


  Scalar latitude_integral_first = integrate_latitudes<Scalar>(data.row(0), quadrature);
  Scalar latitude_integral_left = latitude_integral_first;
  Scalar latitude_integral_right = latitude_integral_first;

  for (eigen::Index i = 0; i < n - 1; ++i) {
    latitude_integral_right =
        integrate_latitudes<Scalar>(data.row(i + 1), quadrature);
    Scalar dl = longitudes[i + 1] - longitudes[i];
    result += 0.5 * (latitude_integral_left + latitude_integral_right) * dl;
    latitude_integral_left = latitude_integral_right;
  }

  Scalar dl = 2.0 * M_PI + longitudes[0] - longitudes[n - 1];
  result += 0.5 * (latitude_integral_first + latitude_integral_right) * dl;

  return result;
}

}  // namespace scatlib
#endif
