/** \file eigen.h
 *
 * Type aliases for Eigen types.
 *
 * @author Simon Pfreundschuh, 2020
 */
#ifndef __SCATTERING_EIGEN__
#define __SCATTERING_EIGEN__

#include <iostream>
#include <type_traits>
#include <memory>

#include <Eigen/CXX11/Tensor>
#include <Eigen/MatrixFunctions>
#include <Eigen/Core>

namespace scattering {
namespace eigen {

using Index = Eigen::Index;

template <int rank>
using IndexArray = std::array<Eigen::DenseIndex, rank>;

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
template <typename Scalar, long int N>
using VectorFixedSize = Eigen::Matrix<Scalar, 1, N, Eigen::RowMajor>;
/** Variable-length vector that doesn't own its data.
 * @tparam Scalar The type used to represent coefficients of the matrix.
 */
template <typename Scalar>
using VectorPtr = std::shared_ptr<Vector<Scalar>>;
template <typename Scalar>
using VectorMap = Eigen::Map<Vector<Scalar>>;
template <typename Scalar>
using VectorMapDynamic =
    Eigen::Map<Vector<Scalar>, 0, Eigen::Stride<1, Eigen::Dynamic>>;
template <typename Scalar>
using ConstVectorMap = Eigen::Map<const Vector<Scalar>>;
template <typename Scalar>
using ConstVectorMapDynamic =
    Eigen::Map<const Vector<Scalar>, 0, Eigen::Stride<1, Eigen::Dynamic>>;
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
template <typename Scalar, long int N>
/** Matrix with fixed number of rows.
 * @tparam Scalar The type used to represent coefficients of the matrix.
 */
using MatrixFixedRows =
    Eigen::Matrix<Scalar, -1, N, (N > 1) ? Eigen::RowMajor : Eigen::ColMajor>;
/** A matrix that doesn't own its data.
 * @tparam Scalar The type used to represent coefficients of the matrix.
 */
template <typename Scalar>
using MatrixMap = Eigen::Map<Matrix<Scalar>>;
template <typename Scalar>
using MatrixMapDynamic =
    Eigen::Map<Matrix<Scalar>,
               Eigen::RowMajor,
               Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
template <typename Scalar>
using ConstMatrixMap = Eigen::Map<const Matrix<Scalar>>;
template <typename Scalar>
using ConstMatrixMapDynamic =
    Eigen::Map<const Matrix<Scalar>,
               Eigen::RowMajor,
               Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
template <typename Scalar>
using MatrixRef = Eigen::Ref<Matrix<Scalar>>;
template <typename Scalar>
using ConstMatrixRef = Eigen::Ref<const Matrix<Scalar>>;

//
// Tensors
//

template <typename Scalar, int rank>
using Tensor = Eigen::Tensor<Scalar, rank, Eigen::RowMajor>;
template <typename Scalar, int rank>
using TensorPtr = std::shared_ptr<Eigen::Tensor<Scalar, rank, Eigen::RowMajor>>;
template <typename Scalar, int rank>
using TensorMap =
    Eigen::TensorMap<Eigen::Tensor<Scalar, rank, Eigen::RowMajor>>;
template <typename Scalar, int rank>
using ConstTensorMap =
    Eigen::TensorMap<const Eigen::Tensor<Scalar, rank, Eigen::RowMajor>>;

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
  using type = Eigen::Map<const Eigen::Matrix<Scalar, rows, cols, options>>;
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

template <typename T, int n_indices>
struct IndexResult {
  using CoeffType = decltype(((T *)nullptr)->operator()({}));
  using Scalar = typename std::remove_reference<typename std::remove_cv<CoeffType>::type>::type;

  static constexpr int rank = T::NumIndices;
  static constexpr bool is_const =
      std::is_const<CoeffType>::value || std::is_const<T>::value;

    using NonConstReturnType = TensorMap<Scalar, rank - n_indices>;
    using ConstReturnType = ConstTensorMap<Scalar, rank - n_indices>;
    using ReturnType = typename std::
      conditional<is_const, ConstReturnType, NonConstReturnType>::type;

  using type =
      typename std::conditional<(n_indices < rank), ReturnType, Scalar>::type;
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

    __attribute__((always_inline)) static inline ReturnType get(T &t,
                               std::array<TensorIndex, n_indices> index_array) {
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
    return ReturnType(t.data() + offset, dimensions_out);
  }
};

template <typename T, int rank_in>
struct TensorIndexer<T, rank_in, rank_in> {
  using Tensor = typename std::remove_reference<T>::type;
  using TensorIndex = typename Tensor::Index;
  using Scalar = typename Tensor::Scalar;

  __attribute__((always_inline)) static inline Scalar get(T &t, std::array<TensorIndex, rank_in> index_array) {
    return t(index_array);
  };
};

// pxx :: export
// pxx :: instance(["4", "scattering::eigen::Tensor<double, 4>"])
// pxx :: instance(["3", "scattering::eigen::Tensor<double, 4>"])
// pxx :: instance(["2", "scattering::eigen::Tensor<double, 4>"])
// pxx :: instance(["1", "scattering::eigen::Tensor<double, 4>"])
// pxx :: instance(["3", "scattering::eigen::Tensor<double, 3>"])
// pxx :: instance(["2", "scattering::eigen::Tensor<double, 3>"])
// pxx :: instance(["1", "scattering::eigen::Tensor<double, 3>"])
// pxx :: instance(["2", "scattering::eigen::Tensor<double, 2>"])
// pxx :: instance(["1", "scattering::eigen::Tensor<double, 2>"])
// pxx :: instance(["1", "scattering::eigen::Tensor<double, 1>"])
template <size_t N, typename T>
    __attribute__((always_inline)) inline auto tensor_index(T &t, std::array<typename T::Index, N> indices) ->
    typename IndexResult<T, N>::type {
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
                         TensorTypeOut &&in,
                         F f,
                         Indices... indices) {
    for (eigen::Index j = 0; j < in.dimension(i); ++j) {
        MapOverDimensionsImpl<N, i + 1>::run(std::forward<TensorTypeIn>(out),
                                             std::forward<TensorTypeOut>(in), f, indices..., j);
    }
  }
};

template <int N>
struct MapOverDimensionsImpl<N, N> {
  template <typename TensorTypeOut,
            typename TensorTypeIn,
            typename F,
            typename... Indices>
  inline static void run(TensorTypeOut &&out,
                         TensorTypeIn &&in,
                         F f,
                         Indices... indices) {
    for (eigen::Index j = 0; j < in.dimension(N - 1); ++j) {
      f(tensor_index<sizeof...(Indices)>(out, {indices...}),
        tensor_index<sizeof...(Indices)>(in, {indices...}));
    }
  }
};

}  // namespace detail

template <int N, typename TensorTypeOut, typename TensorTypeIn, typename F>
void map_over_dimensions(TensorTypeOut &&out, TensorTypeIn &&in, F f) {
  detail::MapOverDimensionsImpl<N>::run(std::forward<TensorTypeOut>(out),
                                        std::forward<TensorTypeIn>(in), f);
}

template <typename T>
auto to_matrix_map(T &t) -> MatrixMap<typename T::Scalar> {
    using Scalar = typename T::Scalar;
  static_assert(T::NumIndices == 2,
                "Tensor must be of rank 2 to be convertible to matrix.");
  return MatrixMap<Scalar>(t.data(), t.dimension(0), t.dimension(1));
}

template <typename T>
auto to_matrix_map(const T &t) -> ConstMatrixMap<typename T::Scalar> {
    using Scalar = typename T::Scalar;
  static_assert(T::NumIndices == 2,
                "Tensor must be of rank 2 to be convertible to matrix.");
  return ConstMatrixMap<Scalar>(t.data(), t.dimension(0), t.dimension(1));
}

template <typename T>
auto to_vector_map(T &t) -> VectorMap<typename T::Scalar> {
    using Scalar = typename T::Scalar;
  static_assert(T::NumIndices == 1,
                "Tensor must be of rank 1 to be convertible to vector.");
  return VectorMap<Scalar>(t.data(), t.dimension(0));
}

template <typename T>
auto to_vector_map(const T &t) -> ConstVectorMap<typename T::Scalar> {
    using Scalar = typename T::Scalar;
  static_assert(T::NumIndices == 1,
                "Tensor must be of rank 1 to be convertible to vector.");
  return ConstVectorMap<Scalar>(t.data(), t.dimension(0));
}

//
// Access to sub-matrix of tensor.
//

namespace detail {
template <typename TensorType>
auto calculate_strides(const TensorType &t) {
  constexpr int rank = TensorType::NumIndices;
  auto dimensions = t.dimensions();
  decltype(dimensions) strides{};
  Index stride = 1;
  for (int i = 0; i < rank; ++i) {
    strides[rank - i - 1] = stride;
    stride *= dimensions[rank - i - 1];
  }
  return strides;
}

}  // namespace detail

// pxx :: export
// pxx :: instance(["1", "3", "scattering::eigen::Tensor<double, 5>", "std::array<int, 3>"])
template <int m,
          int n,
          typename TensorType,
          typename IndexArray = std::array<typename TensorType::Index,
                                           TensorType::NumIndices - 2>>
auto inline get_submatrix(TensorType &t, IndexArray matrix_index) ->
    typename std::conditional<
    !std::is_const<decltype(*(std::declval<TensorType>().data()))>::value
    && !std::is_const<TensorType>::value,
        MatrixMapDynamic<typename TensorType::Scalar>,
        ConstMatrixMapDynamic<typename TensorType::Scalar>>::type
{
  using CoeffType = decltype(*(std::declval<TensorType>().data()));
  using ResultType = typename std::conditional<
      !std::is_const<CoeffType>::value
      && !std::is_const<TensorType>::value,
      MatrixMapDynamic<typename TensorType::Scalar>,
      ConstMatrixMapDynamic<typename TensorType::Scalar>>::type;

  using TensorIndex = typename TensorType::Index;
  constexpr int rank = TensorType::NumIndices;

  // Extend matrix dimensions to tensor dimension
  // to calculate offset.
  auto dimensions_in = t.dimensions();
  std::array<TensorIndex, rank> index{0};
  int dimension_index = 0;
  for (int i = 0; i < rank; ++i) {
    if ((i != m) && (i != n)) {
      index[i] = matrix_index[dimension_index];
      dimension_index++;
    }
  }
  auto offset = dimensions_in.IndexOfRowMajor(index);
  auto strides = detail::calculate_strides(t);

  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> matrix_strides{strides[m], strides[n]};
  auto map = ResultType(t.data() + offset,
                        dimensions_in[m],
                        dimensions_in[n],
                        matrix_strides);
  return map;
}

// pxx :: export
// pxx :: instance(["3", "scattering::eigen::Tensor<double, 5>", "std::array<int, 4>"])
template <int m,
          typename TensorType,
          typename IndexArray = std::array<typename TensorType::Index,
                                           TensorType::NumIndices - 1>>
auto inline get_subvector(TensorType &t, IndexArray vector_index) ->
    typename std::conditional<
    !std::is_const<decltype(*(std::declval<TensorType>().data()))>::value
    && ! std::is_const<TensorType>::value,
        VectorMapDynamic<typename TensorType::Scalar>,
        ConstVectorMapDynamic<typename TensorType::Scalar>>::type
{
  using CoeffType = decltype(*(std::declval<TensorType>().data()));
  using ResultType = typename std::conditional<
      !std::is_const<CoeffType>::value
  && !std::is_const<TensorType>::value,
      VectorMapDynamic<typename TensorType::Scalar>,
      ConstVectorMapDynamic<typename TensorType::Scalar>>::type;

  using TensorIndex = typename TensorType::Index;
  constexpr int rank = TensorType::NumIndices;

  // Extend vector dimensions to tensor dimension
  // to calculate offset.
  auto dimensions_in = t.dimensions();
  std::array<TensorIndex, rank> index{0};
  int dimension_index = 0;
  for (int i = 0; i < rank; ++i) {
    if (i != m) {
      index[i] = vector_index[dimension_index];
      dimension_index++;
    }
  }
  auto offset = dimensions_in.IndexOfRowMajor(index);
  auto strides = detail::calculate_strides(t);

  Eigen::Stride<1, Eigen::Dynamic> vector_strides{1, strides[m]};
  auto map = ResultType(t.data() + offset,
                        1,
                        dimensions_in[m],
                        vector_strides);
  return map;
}

////////////////////////////////////////////////////////////////////////////////
// Dimension counter
////////////////////////////////////////////////////////////////////////////////

/** Tensor index counter to loop over elements.
 *
 * The dimensions counter is a helper class that allows looping over the
 * indices in a tensor in a single loop. It loops over all elements in the
 * tensor in row-major order, meaning that the last index increased with
 * every increment of the counter. The current tensor index can be
 * accessed through the coordinates member.
 */
template <int rank>
struct DimensionCounter {
  /** Create counter
   * @param dims Array containing the dimension of the tensor to loop over.
   */
  DimensionCounter(std::array<Eigen::DenseIndex, rank> dims) {
    dimensions = dims;
  }

  /// Increment the counter.
  DimensionCounter &operator++() {
    for (int i = rank - 1; i >= 0; i--) {
      coordinates[i]++;
      if (coordinates[i] == dimensions[i]) {
        coordinates[i] = 0;
        if (i == 0) {
          exhausted_ = true;
        }
      } else {
        break;
      }
    }
    return *this;
  }

  /// Have all elements been looped over?
  operator bool() { return !exhausted_; }

  bool exhausted_ = false;
  /// The index of the current tensor element.
  std::array<Eigen::DenseIndex, rank> coordinates{0};
  std::array<Eigen::DenseIndex, rank> dimensions{0};
};

template <int rank>
std::ostream &operator<<(std::ostream &out, const DimensionCounter<rank> &c) {
  out << "Dimension counter: ";
  for (int i = 0; i < rank - 1; ++i) {
    out << c.coordinates[i] << ", ";
  }
  out << c.coordinates[rank - 1] << std::endl;
  return out;
}

template <typename Scalar, typename ... Types>
Tensor<Scalar, sizeof...(Types)> zeros(Types ... dimensions) {
    constexpr int  rank = sizeof...(Types);
    std::array<Index, rank> dimension_array({dimensions ...});
    return Tensor<Scalar, sizeof...(Types)>(dimension_array).setZero();
}

template <int ... dims, typename Scalar, int rank>
Tensor<Scalar, rank + sizeof ... (dims)> unsqueeze(const Tensor<Scalar, rank> &tensor) {
    constexpr int new_rank = rank + sizeof ... (dims);
    auto dimensions = tensor.dimensions();
    std::array<Index, new_rank> new_dimensions;
    std::array<Index, sizeof ... (dims)> trivial_dimensions{dims ...};
    std::sort(trivial_dimensions.begin(), trivial_dimensions.end());
    auto dimension_iterator = dimensions.begin();
    auto trivial_dimension_iterator = trivial_dimensions.begin();

    for (int i = 0; i < new_rank; ++i) {
        if (i == *trivial_dimension_iterator) {
            new_dimensions[i] = 1;
            trivial_dimension_iterator++;
        } else {
            new_dimensions[i] = *dimension_iterator;
            dimension_iterator++;
        }
    }
    return tensor.reshape(new_dimensions);
}

template <typename Scalar, int rank>
    Tensor<Scalar, rank> cycle_dimensions(const Tensor<Scalar, rank> &t) {
    std::array<Index, rank> dimensions = {};
    for (int i = 0; i < rank; ++i) {
        dimensions[i] = (i + 1) % rank;
    }
    return t.shuffle(dimensions);
}

template <typename TensorType>
struct CopyGenerator {
  static constexpr int rank = TensorType::NumIndices;
  using Scalar = typename TensorType::Scalar;
  CopyGenerator(const TensorType &from_) : from(from_){};

  Scalar operator()(const std::array<Index, rank> &coordinates) const {
    for (size_t i = 0; i < rank; ++i) {
      if (coordinates[i] >= from.dimension(i)) {
        return 0.0;
      }
    }
    return from(coordinates);
  }

  const TensorType &from;
};

template <typename TensorType1, typename TensorType2>
void copy(TensorType1 &dest, const TensorType2 &source) {
    dest = dest.generate(CopyGenerator<TensorType2>(source));
}

template <typename VectorType1, typename VectorType2>
bool equal(const VectorType1 &left, const VectorType2 &right) {
  if (left.size() != right.size()) {
    return false;
  }
  return left == right;
}

template <typename Scalar>
auto colatitudes(const Vector<Scalar>& input) {
    return input.unaryExpr([](Scalar x) { return -1.0 * cos(x);});
}

template<Index n, typename TensorType>
std::array<Index, n> get_dimensions(const TensorType &t) {
    std::array<Index, n> result{};
    for (Index i = 0; i < n; ++i) {
        if (i < TensorType::NumIndices) {
            result[i] = t.dimension(i);
        } else {
            result[i] = 0;
        }
    }
    return result;
}


}  // namespace eigen
}  // namespace scattering

#endif
