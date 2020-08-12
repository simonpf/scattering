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
template <typename Scalar>
using ConstVectorMap = Eigen::Map<const Vector<Scalar>>;
template <typename Scalar>
using VectorRef = Eigen::Ref<Vector<Scalar>>;
template <typename Scalar>
using ConstVectorRef = Eigen::Ref<const Vector<Scalar>>;

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
using MatrixFixedRows = Eigen::Matrix<Scalar, -1, N, (N>1) ? Eigen::RowMajor : Eigen::ColMajor>;
/** A matrix that doesn't own its data.
 * @tparam Scalar The type used to represent coefficients of the matrix.
 */
template <typename Scalar>
using MatrixMap = Eigen::Map<Matrix<Scalar>>;
template <typename Scalar>
using ConstMatrixMap = Eigen::Map<const Matrix<Scalar>>;

//
// Tensors
//

template <typename Scalar, int rank>
using Tensor = Eigen::Tensor<Scalar, rank, Eigen::RowMajor>;
template <typename Scalar, int rank>
using TensorMap = Eigen::TensorMap<Eigen::Tensor<Scalar, rank, Eigen::RowMajor>>;
template <typename Scalar, int rank>
using ConstTensorMap = Eigen::TensorMap<const Eigen::Tensor<Scalar, rank, Eigen::RowMajor>>;

//
// Tensor map
//

namespace detail {
template <size_t i>
struct MapOverDimensionsImpl {
    template <typename TensorTypeOut, typename TensorTypeIn, typename f, typename ... Indices>
    inline static void run(TensorTypeIn &out, const TensorTypeOut &in, Indices ... indices) {
    int current_dimension = sizeof...(indices);
    for (eigen::Index j = 0; j < in.dimension(current_dimension); ++j) {
      MapOverDimensionsImpl::run(out, in, indices..., j);
    }
  }
};

template <>
struct MapOverDimensionsImpl<0> {
    template <typename TensorTypeOut, typename TensorTypeIn, typename f, typename ... Indices>
        inline static void run(TensorTypeOut out, TensorTypeIn in, Indices ... indices) {
    int current_dimension = sizeof...(indices);
    for (eigen::Index j = 0; j < in.dimension(current_dimension); ++j) {
      out(indices...) = f(indices...);
    }
  }
};

template <typename T> struct TensorTransformTrait;

template <typename TensorIn, typename TensorOut>
    struct TensorTransformTrait<std::function<TensorOut(TensorIn)>> {
    using TensorTypeOut = TensorOut;
    static constexpr int rank_out = TensorOut::NumIndices;
    using TensorTypeIn = TensorIn;
    static constexpr int rank_in = TensorIn::NumIndices;
};

}  // namespace detail

template <typename TensorTypeOut, typename TensorTypeIn, typename f>
void map_over_dimensions(TensorTypeOut &out, const TensorTypeIn &in) {
  static constexpr int ranks_in = TensorTypeIn::NumIndices;
  static constexpr int ranks_to_map_over =
      ranks_in - detail::TensorTransformTrait<f>::rank_in;
  detail::MapOverDimensionsImpl<ranks_to_map_over>::run(out, in);
}

}  // namespace detail
}  // namespace scatlib

#endif
