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

template <typename T>
struct ResultTypeHelper;

template <typename T>
struct ResultTypeHelper<Eigen::TensorMap<T, Eigen::RowMajor>> {
    template <int rank>
    using type = typename ResultTypeHelper<T>::template type<rank>;
};

template <typename Scalar, int NumIndices, int Options, typename IndexType>
struct ResultTypeHelper<Eigen::Tensor<Scalar, NumIndices, Options, IndexType>> {
  template <int rank>
  using type = Eigen::TensorMap<
      Eigen::Tensor<Scalar, NumIndices - rank, Options, IndexType>>;
};

//
// Template helper to create map to sub tensor.
//

template <typename T, int rank_in, int n_indices>
struct TensorIndexer {
  using Tensor = typename std::remove_reference<T>::type;
  using TensorIndex = typename Tensor::Index;
  using ReturnType = typename Map<
      typename ResultTypeHelper<Tensor>::template type<n_indices>>::type;
  using Scalar = typename Tensor::Scalar;
  static constexpr int rank_out = rank_in - n_indices;


  template <typename InputType>
  static inline ReturnType get(InputType &t,
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

  template <typename InputType>
  static inline Scalar get(InputType &t, std::array<TensorIndex, rank_in> index_array) {
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
// pxx :: instance(["Eigen::TensorMap<Eigen::Tensor<double, 4, Eigen::RowMajor>, Eigen::RowMajor>", "Eigen::Index", "1"])
template <typename T, typename Index, size_t N>
auto tensor_index(T &t, std::array<Index, N> indices) {
    return TensorIndexer<T, T::NumIndices, N>::get(t, indices);
}

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
        out({indices...}) = f({indices...});
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
