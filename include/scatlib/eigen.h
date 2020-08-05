/** \file eigen.h
 *
 * Type aliases for Eigen types.
 *
 * @author Simon Pfreundschuh, 2020
 */
#ifndef __SCATLIB_EIGEN__
#define __SCATLIB_EIGEN__

#include <Eigen/CXX11/Tensor>
#include <Eigen/Core>

namespace scatlib {
namespace eigen {

using Index = Eigen::Index;

//
// Vectors
//

/** Variable-length vector.
 * @tparam Scalar The type used to represent coefficients of the matrix.
 */
template <typename Scalar>
using Vector = Eigen::Matrix<Scalar, 1, -1, Eigen::RowMajor>;
/** Fixed-length vector.
 * @tparam Scalar The type used to represent coefficients of the matrix.
 */
template <typename Scalar, size_t N>
using VectorFixedSize = Eigen::Matrix<Scalar, 1, N, Eigen::RowMajor>;
/** Variable-length vector that doesn't own its data.
 * @tparam Scalar The type used to represent coefficients of the matrix.
 */
template <typename Scalar>
using VectorMap = Eigen::Map<Vector<Scalar>>;

//
// Matrices
//

/** A variable-size matrix containing coefficients of type Scalar.
 * @tparam Scalar The type used to represent coefficients of the matrix.
 */
template <typename Scalar>
using Matrix = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;
template <typename Scalar, size_t N>
/** Matrix with fixed number of rows.
 * @tparam Scalar The type used to represent coefficients of the matrix.
 */
using MatrixFixedRows = Eigen::Matrix<Scalar, -1, N, Eigen::RowMajor>;
/** A matrix that doesn't own its data.
 * @tparam Scalar The type used to represent coefficients of the matrix.
 */
template <typename Scalar>
using MatrixMap = Eigen::Map<Matrix<Scalar>>;

//
// Tensors
//

template <typename Scalar, size_t rank>
using Tensor = Eigen::Tensor<Scalar, rank, Eigen::RowMajor>;

}  // namespace eigen
}  // namespace scatlib

#endif
