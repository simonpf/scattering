#ifndef __SCATTERING_UTILS_MATH__
#define __SCATTERING_UTILS_MATH__

#include <cmath>

namespace scattering {
namespace math {
template<typename Scalar> bool equal(Scalar a,
                                     Scalar b,
                                     Scalar epsilon = 1e-6) {
  return std::abs(a - b) <=
         ((std::abs(a) > std::abs(b) ? std::abs(b) : std::abs(a)) * epsilon);
}

template<typename Scalar> bool small(Scalar a, Scalar epsilon = 1e-6) {
  return equal(a, 0.0, epsilon);
}

template <typename Scalar>
Scalar save_acos(Scalar a, Scalar epsilon = 1e-6) {
  if (equal(a, 1.0, epsilon)) {
    return 0.0;
  }
  if (equal(a, -1.0, epsilon)) {
    return M_PI;
  }
  return acos(a);
}

}  // namespace math
}  // namespace scattering
# endif
