/** \file eigen.h
 *
 * Type alias for Eigen types.
 *
 * @author Simon Pfreundschuh, 2020
 */
#ifndef __SCATLIB_EIGEN__
#define __SCATLIB_EIGEN__

#include <Eigen/CXX11/Tensor>
#include <Eigen/Core>

namespace scatlib {
namespace eigen {
template <typename Scalar>
using Matrix = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;
template <typename Scalar, size_t N>
using MatrixFixedRows = Eigen::Matrix<Scalar, -1, N, Eigen::RowMajor>;
template <typename Scalar>
using Vector = Eigen::Matrix<Scalar, 1, -1, Eigen::RowMajor>;
template <typename Scalar, size_t N>
using VectorFixedSize = Eigen::Matrix<Scalar, 1, N, Eigen::RowMajor>;
}  // namespace eigen
}  // namespace scatlib

#endif
