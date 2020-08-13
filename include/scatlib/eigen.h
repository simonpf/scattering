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
#include <iostream>
#include <type_traits>

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
// General map type.
//

template <typename Tensor>
    struct Map {
        using type = Eigen::TensorMap<Tensor>;
    };

template <typename Scalar, int rows, int cols, int options>
    struct Map<Eigen::Matrix<Scalar, rows, cols, options>> {
    using type = Eigen::Map<Eigen::Matrix<Scalar, rows, cols, options>>;
};

template <typename Scalar, int rows, int cols, int options>
    struct Map<const Eigen::Matrix<Scalar, rows, cols, options>> {
    using type = Eigen::Map <const Eigen::Matrix<Scalar, rows, cols, options>>;
};

template <typename Derived>
    struct Map<Eigen::TensorMap<Derived>> {
    using type = typename Map<Derived>::type;
};

template <typename Derived>
    struct Map<Eigen::Map<Derived>> {
    using type = typename Map<Derived>::type;
};

template <typename Derived>
    struct Map<const Eigen::TensorMap<Derived>> {
    using type = typename Map<Derived>::type;
};

template <typename Derived>
    struct Map<const Eigen::Map<Derived>> {
    using type = typename Map<Derived>::type;
};

////////////////////////////////////////////////////////////////////////////////
// Tensor indexing
////////////////////////////////////////////////////////////////////////////////

//
// Helper trait to reduce rank.
//
template <typename T> int foo(T *t);

template <typename T, int n_indices>
struct IndexResult {

    using CoeffType = decltype(((T *) nullptr)->operator()({}));
    using Scalar = typename std::remove_cvref<CoeffType>::type;

    static constexpr int rank = T::NumIndices;
    static constexpr bool is_const = std::is_const<CoeffType>::value || std::is_const<T>::value;

    using NonConstReturnType = Eigen::TensorMap<
        Eigen::Tensor<Scalar, rank - n_indices, Eigen::RowMajor, Eigen::Index>,
        Eigen::RowMajor>;
    using ConstReturnType = Eigen::TensorMap<
        const Eigen::
            Tensor<Scalar, rank - n_indices, Eigen::RowMajor, Eigen::Index>,
        Eigen::RowMajor>;
    using ReturnType = typename std::conditional<is_const, ConstReturnType, NonConstReturnType>::type;

    using type = typename std::conditional<(n_indices < rank), ReturnType, Scalar>::type;
};

//
// Template helper to create map to sub tensor.
//

template <typename T, int rank_in, int n_indices>
struct TensorIndexer {
  using Tensor = typename std::remove_reference<T>::type;
  using TensorIndex = typename Tensor::Index;
  using ReturnType = typename IndexResult<Tensor, n_indices>::type;
  using Scalar = typename Tensor::Scalar;
  static constexpr int rank_out = rank_in - n_indices;


  static inline ReturnType get(T &t,
                 std::array<TensorIndex, n_indices> index_array)
        {
    auto dimensions_in = t.dimensions();
    std::array<TensorIndex, rank_in> index{};
    for (int i = 0; i < n_indices; ++i) {
      index[i] = index_array[i];
    }
    Eigen::DSizes<TensorIndex, rank_out> dimensions_out{};
    for (int i = 0; i < rank_out; ++i) {
      dimensions_out[i] = dimensions_in[n_indices + i];
    }
    auto offset = dimensions_in.IndexOfRowMajor(index);
    std::cout << "offset: " << offset << std::endl;
    for(int i =0; i < rank_in; ++i) {
        std::cout << index[i] << " / ";
    }
    std::cout << std::endl;
    for(int i =0; i < rank_out; ++i) {
        std::cout << dimensions_out[i] << " :: ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    return ReturnType(t.data() + offset, dimensions_out);
  }
};

template <typename T, int rank_in>
struct TensorIndexer<T, rank_in, rank_in> {
  using Tensor = typename std::remove_reference<T>::type;
  using TensorIndex = typename Tensor::Index;
  using Scalar = typename Tensor::Scalar;

  static inline Scalar get(T &t, std::array<TensorIndex, rank_in> index_array) {
    return t(index_array);
  };
};

// pxx :: export
// pxx :: instance(["Eigen::TensorMap<Eigen::Tensor<double, 4, Eigen::RowMajor>, Eigen::RowMajor>", "Eigen::Index", "1"])
// pxx :: instance(["Eigen::TensorMap<Eigen::Tensor<double, 4, Eigen::RowMajor>, Eigen::RowMajor>", "Eigen::Index", "2"])
// pxx :: instance(["Eigen::TensorMap<Eigen::Tensor<double, 4, Eigen::RowMajor>, Eigen::RowMajor>", "Eigen::Index", "3"])
// pxx :: instance(["Eigen::TensorMap<Eigen::Tensor<double, 4, Eigen::RowMajor>, Eigen::RowMajor>", "Eigen::Index", "4"])
// pxx :: instance(["Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>, Eigen::RowMajor>", "Eigen::Index", "1"])
// pxx :: instance(["Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>, Eigen::RowMajor>", "Eigen::Index", "2"])
// pxx :: instance(["Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>, Eigen::RowMajor>", "Eigen::Index", "3"])
// pxx :: instance(["Eigen::TensorMap<Eigen::Tensor<double, 2, Eigen::RowMajor>, Eigen::RowMajor>", "Eigen::Index", "1"])
// pxx :: instance(["Eigen::TensorMap<Eigen::Tensor<double, 2, Eigen::RowMajor>, Eigen::RowMajor>", "Eigen::Index", "2"])
// pxx :: instance(["Eigen::TensorMap<Eigen::Tensor<double, 1, Eigen::RowMajor>, Eigen::RowMajor>", "Eigen::Index", "1"])
template <typename T, typename Index, size_t N>
auto tensor_index(T &t, std::array<Index, N> indices)
    -> typename IndexResult<T, N>::type
{
    return TensorIndexer<T, T::NumIndices, N>::get(t, indices);
}

//
// Tensor map
//


namespace detail {

template <int N, int i = 0>
struct MapOverDimensionsImpl {
  template <typename TensorTypeOut,
            typename TensorTypeIn,
            typename F,
            typename... Indices>
  inline static void run(TensorTypeIn &&out,
                         const TensorTypeOut &&in,
                         F f,
                         Indices... indices) {
    for (eigen::Index j = 0; j < in.dimension(i); ++j) {
      MapOverDimensionsImpl<N, i + 1>::run(out, in, f, indices..., j);
    }
  }
};

template <int N>
struct MapOverDimensionsImpl<N, N - 1> {
  template <typename TensorTypeOut,
            typename TensorTypeIn,
            typename F,
            typename... Indices>
  inline static void run(TensorTypeOut &&out,
                         TensorTypeIn &&in,
                         F f,
                         Indices... indices) {
    for (eigen::Index j = 0; j < in.dimension(N - 1); ++j) {
        f(tensor_index(out, {indices...}), tensor_index(in, {indices...}));
    }
  }
};

}  // namespace detail

template <int N, typename TensorTypeOut, typename TensorTypeIn, typename F>
void map_over_dimensions(TensorTypeOut &&out, TensorTypeIn &&in, F f) {
  detail::MapOverDimensionsImpl<N>::run(out, in, f);
}

template <typename T>
auto  to_matrix_map(T &t)
    -> MatrixMap<typename T::Scalar> {
    static_assert(T::NumIndices == 2,
                       "Tensor must be of rank 2 to be convertible to matrix.");
    MatrixMap(t.data(), t.dimension(0), t.dimensions(1));
}

template <typename T>
    auto  to_matrix_map(const T &t)
    -> ConstMatrixMap<typename T::Scalar> {
    static_assert(T::NumIndices == 2,
                  "Tensor must be of rank 2 to be convertible to matrix.");
    ConstMatrixMap(t.data(), t.dimension(0), t.dimensions(1));
}

template <typename T>
    auto  to_vector_map(T &t)
    -> VectorMap<typename T::Scalar> {
    static_assert(T::NumIndices == 1,
                       "Tensor must be of rank 1 to be convertible to vector.");
    VectorMap(t.data(), t.dimension(0));
}

template <typename T>
    auto  to_vector_map(const T &t)
    -> ConstVectorMap<typename T::Scalar> {
    static_assert(T::NumIndices == 1,
                       "Tensor must be of rank 1 to be convertible to vector.");
    ConstVectorMap(t.data(), t.dimension(0));
}


}  // namespace detail
}  // namespace scatlib

#endif
