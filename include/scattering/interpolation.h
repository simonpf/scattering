/** \file interpolation.h
 *
 * Generic interpolation method for Eigen tensors on regular
 * grids.
 *
 * @author Simon Pfreundschuh, 2020
 */
#ifndef __SCATTERING_INTERPOLATION__
#define __SCATTERING_INTERPOLATION__

#include <scattering/eigen.h>

#include <algorithm>
#include <utility>
#include <chrono>
#include <iostream>
#include <type_traits>
#include <vector>

namespace scattering {
namespace detail {

//
// Interpolation helper.
//

template <typename Derived>
struct is_vector {
  static constexpr bool value =
      (Derived::RowsAtCompileTime == 1) || (Derived::ColsAtCompileTime == 1);
};

template <typename T, Eigen::Index N>
struct InterpolationResult {
    using Scalar = typename T::Scalar;
  static constexpr int rank = T::NumIndices;
    using ReturnType = eigen::Tensor<Scalar, rank - N>;

    using type = typename std::conditional<(N < rank), ReturnType, Scalar>::type;
};

template <typename Derived, Eigen::Index N>
struct InterpolationResult<Eigen::Map<Derived>, N> {
  using Scalar = typename Derived::Scalar;

  template <typename T>
  static eigen::Vector<Scalar> type_helper(
      T*,
      typename std::enable_if<(!is_vector<T>::value) && (N == 1)>::type* = 0) {
    return {};
  }
  template <typename T>
  static Scalar type_helper(
      const T*,
      typename std::enable_if<is_vector<T>::value || (N == 2)>::type* = 0) {
    return nullptr;
  }

  using type = decltype(type_helper((Derived*)nullptr));
};

template <typename Tensor, Eigen::Index N, Eigen::Index I = 0>
struct Interpolator {
  using Result = typename InterpolationResult<Tensor, N>::type;
  using Scalar = typename Tensor::Scalar;
  using Real = decltype(std::real(std::declval<Scalar>()));
  using Index = typename Tensor::Index;

     static inline Result compute(
      const Tensor& tensor,
      const eigen::VectorFixedSize<Real, N> &weights,
      const eigen::VectorFixedSize<Index, N> &indices,
      const eigen::VectorFixedSize<Index, I> &offsets = {})

        {
    eigen::VectorFixedSize<Index, N> indices_new{indices};
    for (Index i = 0; i < I; ++i) {
      indices_new[i] += offsets[i];
    }

    eigen::VectorFixedSize<Index, I + 1> offsets_new{};
    for (Index i = 0; i < I; ++i) {
      offsets_new[i] = offsets[i];
    }
    offsets_new[I] = 0;

    Real w = weights[I];
    Result t = Interpolator<Tensor, N, I + 1>::compute(tensor,
                                                       weights,
                                                       indices,
                                                       offsets_new);
    if (w < 1.0) {
      offsets_new[I] = 1;
      t = w * t;
      t += static_cast<Real>(1.0 - w) *
           Interpolator<Tensor, N, I + 1>::compute(tensor,
                                                   weights,
                                                   indices,
                                                   offsets_new);
    }
    return t;
  }
};

template <typename Tensor, Eigen::Index N>
struct Interpolator<Tensor, N, N> {
  using Result = typename eigen::IndexResult<const Tensor, N>::type;
  using Scalar = typename Tensor::Scalar;
  using Real = decltype(std::real(std::declval<Scalar>()));
  using Index = typename Tensor::Index;

   static inline Result compute(
      const Tensor& tensor,
      const eigen::VectorFixedSize<Real, N> &/*weights*/,
      const eigen::VectorFixedSize<Index, N> &indices,
      const eigen::VectorFixedSize<Index, N> &offsets = {}) {
    std::array<Index, N> indices_new;
    for (Index i = 0; i < N; ++i) {
      indices_new[i] = indices[i] + offsets[i];
    }
    return eigen::tensor_index(tensor, indices_new);
  }
};


template <typename Derived>
    struct Interpolator<Eigen::Map<Derived>, 2, 2> {
    using Matrix = Eigen::Map<Derived>;
    using Scalar = typename Derived::Scalar;
    using Real = decltype(std::real(std::declval<Scalar>()));

    __attribute__((always_inline)) static inline Scalar compute(
        const Matrix& matrix,
        const eigen::VectorFixedSize<Real, 2> &/*weights*/,
        const eigen::VectorFixedSize<Eigen::Index, 2> &indices,
        const eigen::VectorFixedSize<Eigen::Index, 2> &offsets = {}) {
        return matrix(indices[0] + offsets[0], indices[1] + offsets[1]);
    }
};


template <typename Derived>
    struct Interpolator<Eigen::Map<Derived>, 1, 1> {
    using Vector = Eigen::Map<Derived>;
    using Matrix = Eigen::Map<Derived>;
    using Scalar = typename Derived::Scalar;
    using Real = decltype(std::real(std::declval<Scalar>()));

    template <typename T>
    __attribute__((always_inline)) static inline auto compute(
        const Eigen::Map<T>& matrix,
        const eigen::VectorFixedSize<Real, 1> &/*weights*/,
        const eigen::VectorFixedSize<Eigen::Index, 1> &indices,
        const eigen::VectorFixedSize<Eigen::Index, 1> &offsets = {},
        typename std::enable_if<!is_vector<T>::value>::type * = 0) {
        return matrix.row(indices[0] + offsets[0]);
    }

    template <typename T>
        __attribute__((always_inline)) static inline Scalar compute(
        const Eigen::Map<T>& vector,
        const eigen::VectorFixedSize<Real, 1> &/*weights*/,
        const eigen::VectorFixedSize<Eigen::Index, 1> &indices,
        const eigen::VectorFixedSize<Eigen::Index, 1> &offsets = {},
        typename std::enable_if<is_vector<T>::value>::type * = 0) {
        return vector[indices[0] + offsets[0]];
    }
};

/* template <typename Scalar, int rows, int cols, int Options,> */
/*     struct Interpolator<Eigen::Matrix<Scalar, rows, cols, Options>, 2, 2> { */
/*     using Matrix = Eigen::ConstMatrix<Scalar, 1, -1, Options>; */

/*     template <typename VectorType> */
/*         static inline Scalar compute( */
/*             const Matrix& vector, */
/*             const eigen::VectorFixedSize<Scalar, 1> &/\*weights*\/, */
/*             const eigen::VectorFixedSize<Eigen::Index, 1> &indices, */
/*             const eigen::VectorFixedSize<Eigen::Index, 1> &offsets = {}) { */
/*         return vector(indices[0] + offsets[0], indices[0] + offsets[0]); */
/*     } */
/* }; */

/* template <typename Scalar, int Options> */
/* struct Interpolator<Eigen::Matrix<Scalar, 1, -1, Options>, 1, 1> { */
/*   using Vector = Eigen::Matrix<Scalar, 1, -1, Options>; */

/*   template <typename VectorType> */
/*   static inline Scalar compute( */
/*       const VectorType& vector, */
/*       const eigen::VectorFixedSize<Scalar, 1> &/\*weights*\/, */
/*       const eigen::VectorFixedSize<Eigen::Index, 1> &indices, */
/*       const eigen::VectorFixedSize<Eigen::Index, 1> &offsets = {}) { */
/*     return vector[indices[0] + offsets[0]]; */
/*   } */
/* }; */

/* template <typename Scalar, int Options> */
/* struct Interpolator<Eigen::Matrix<Scalar, -1, 1, Options>, 1, 1> { */
/*   using Vector = Eigen::Matrix<Scalar, -1, 1, Options>; */

/*   template <typename VectorType> */
/*   static inline Scalar compute( */
/*       const VectorType& vector, */
/*       const eigen::VectorFixedSize<Scalar, 1> &/\*weights*\/, */
/*       const eigen::VectorFixedSize<Eigen::Index, 1> &indices, */
/*       const eigen::VectorFixedSize<Eigen::Index, 1> &offsets = {}) { */
/*     return vector[indices[0] + offsets[0]]; */
/*   } */
/* }; */

/* template <typename Scalar, int Options> */
/* struct Interpolator<Eigen::Map<const Eigen::Matrix<Scalar, 1, -1, Options>>, 1, 1> { */
/* using Vector = Eigen::Map<const Eigen::Matrix<Scalar, 1, -1, Options>>; */

/*         static inline Scalar compute( */
/*             const Vector& vector, */
/*             eigen::VectorFixedSize<Scalar, 1> /\*weights*\/, */
/*             eigen::VectorFixedSize<Eigen::Index, 1> indices, */
/*             eigen::VectorFixedSize<Eigen::Index, 1> offsets = {}) { */
/*         return vector[indices[0] + offsets[0]]; */
/*     } */
/* }; */

/* template <typename Scalar, int Options> */
/*     struct Interpolator<Eigen::Map<const Eigen::Matrix<Scalar, -1, 1, Options>>, 1, 1> { */
/*     using Vector = Eigen::Map<const Eigen::Matrix<Scalar, -1, 1, Options>>; */

/*     static inline Scalar compute( */
/*         const Vector& vector, */
/*         eigen::VectorFixedSize<Scalar, 1> /\*weights*\/, */
/*         eigen::VectorFixedSize<Eigen::Index, 1> indices, */
/*         eigen::VectorFixedSize<Eigen::Index, 1> offsets = {}) { */
/*         return vector[indices[0] + offsets[0]]; */
/*     } */
/* }; */

/* template <typename Derived, Eigen::Index N> */
/* struct Interpolator<Eigen::Map<Derived>, N> { */
/*   template <typename... Types> */
/*   static inline auto compute(const Types& ... types) { */
/*       return Interpolator<std::remove_cvref_t<Derived>, N>::compute(types ...); */
/*   } */
/* }; */

//
// Calculating interpolation weights.
//

template <typename Scalar>
eigen::Vector<Eigen::Index> indirect_sort(const eigen::Vector<Scalar>& v) {
  eigen::Vector<Eigen::Index> indices;
  indices.setLinSpaced(v.size(), 0, v.size() - 1);

  auto comp = [&v](size_t i, size_t j) { return v[i] < v[j]; };
  std::sort(indices.begin(), indices.end(), comp);
  return indices;
}

template <typename Scalar>
using WeightIndexPair =
    std::pair<eigen::Vector<Scalar>, eigen::Vector<Eigen::Index>>;

template <typename WeightVector,
          typename IndexVector,
          typename GridVector,
          typename PositionVector>
void calculate_weights(WeightVector&& weights,
                       IndexVector&& indices,
                       const GridVector& grid,
                       const PositionVector& positions,
                       bool extrapolate=false) {
  using Scalar = typename std::remove_reference<WeightVector>::type::Scalar;

  for (int i = 0; i < positions.size(); ++i) {
    auto p = positions[i];
    auto f = std::lower_bound(grid.begin(), grid.end(), p);
    if (extrapolate || (f != grid.end())) {
      // p is within grid limits.
      if (etrapolate || (f != grid.begin())) {
          indices[i] = f - grid.begin();
          if (*f != p) {
              indices[i] -= 1;
          }
          Scalar l = grid[indices[i]];
          if (l == p) {
              weights[i] = 1.0;
          } else {
              Scalar r = grid[indices[i] + 1];
              weights[i] = (r - p) / (r - l);
          }
      } else {
      // p is left of lower limit.
      indices[i] = 0;
      weights[i] = 1.0;
      }
    } else {
      // p is right of upper limit.
      indices[i] = grid.size() - 1;
      weights[i] = 1.0;
    }
  }
}

template <typename Scalar>
WeightIndexPair<Scalar> calculate_weights(
    const eigen::Vector<Scalar>& grid,
    const eigen::Vector<Scalar>& positions) {
  if (positions.size() == 0) {
    return std::make_pair<eigen::Vector<Scalar>, eigen::Vector<Eigen::Index>>(
        {},
        {});
  }

  eigen::Vector<Scalar> weights =
      eigen::Vector<Scalar>::Constant(positions.size(), 1.0);
  eigen::Vector<eigen::Index> indices =
      eigen::Vector<eigen::Index>::Zero(positions.size());

  if (grid.size() == 1) {
    return std::make_pair(weights, indices);
  }

  calculate_weights(weights, indices, grid, positions);

  return std::make_pair(weights, indices);
}

}

// pxx :: export
// pxx :: instance(["Eigen::Tensor<float, 4, Eigen::RowMajor>", "3", "float"])
// pxx :: instance(["Eigen::Tensor<float, 5, Eigen::RowMajor>", "3", "float"])
// pxx :: instance(["Eigen::Tensor<float, 6, Eigen::RowMajor>", "3", "float"])
// pxx :: instance(["Eigen::Tensor<float, 7, Eigen::RowMajor>", "3", "float"])
// pxx :: instance(["Eigen::Tensor<float, 3, Eigen::RowMajor>", "2", "float"])
// pxx :: instance(["Eigen::Tensor<float, 4, Eigen::RowMajor>", "2", "float"])
// pxx :: instance(["Eigen::Tensor<float, 5, Eigen::RowMajor>", "2", "float"])
// pxx :: instance(["Eigen::Tensor<float, 6, Eigen::RowMajor>", "2", "float"])
// pxx :: instance(["Eigen::Tensor<float, 7, Eigen::RowMajor>", "2", "float"])
// pxx :: instance(["Eigen::Tensor<float, 2, Eigen::RowMajor>", "1", "float"])
// pxx :: instance(["Eigen::Tensor<float, 3, Eigen::RowMajor>", "1", "float"])
// pxx :: instance(["Eigen::Tensor<float, 4, Eigen::RowMajor>", "1", "float"])
// pxx :: instance(["Eigen::Tensor<float, 5, Eigen::RowMajor>", "1", "float"])
// pxx :: instance(["Eigen::Tensor<float, 6, Eigen::RowMajor>", "1", "float"])
// pxx :: instance(["Eigen::Tensor<float, 7, Eigen::RowMajor>", "1", "float"])
//
/** Interpolate tensor using given weights and indices.
 *
 * Piece-wise linear interpolation of the given tensor along its first
 * dimensions.
 *
 * @tparam degree Along how many dimensions the interpolation is performed.
 * @param tensor Rank-k tensor to interpolate.
 * @param weights Vector containing the interpolation weights giving the
 * weighting of the left boundary of the interpolation domain.
 * @param indices Vector containing the indices of the left boundaries of the
 * interpolation domain.
 * @return Rank-(k - degree) tensor containing the result.
 */
template <typename Tensor, size_t degree, typename Scalar>
    __attribute__((always_inline)) inline typename detail::InterpolationResult<Tensor, degree>::type interpolate(
    const Tensor& tensor,
    Eigen::Ref<const eigen::VectorFixedSize<Scalar, degree>>
        weights,
    Eigen::Ref<const eigen::VectorFixedSize<Eigen::Index, degree>>
        indices) {
  return detail::Interpolator<Tensor, degree>::compute(tensor,
                                                       weights,
                                                       indices);
}

// pxx :: export
// pxx :: instance(["Eigen::Tensor<double, 5, Eigen::RowMajor>", "3", "Eigen::VectorXd"])
//
/** Regular grid interpolator.
 *
 * Piecewise-linear interpolator on regular grids.
 *
 * @tparam Tensor The Eigen tensor type to interpolate.
 * @tparam degree Along how many dimensions to interpolate.
 */
template <typename Tensor, size_t degree, typename Vector>
class RegularGridInterpolator {
 public:
  using Scalar = typename Vector::Scalar;
  using WeightVector = eigen::VectorFixedSize<Scalar, degree>;
  using IndexVector = eigen::VectorFixedSize<Eigen::Index, degree>;
  using WeightMatrix = eigen::MatrixFixedRows<Scalar, degree>;
  using IndexMatrix = eigen::MatrixFixedRows<Eigen::Index, degree>;
  using InterpolationWeights = std::pair<WeightMatrix, IndexMatrix>;

  /** Sets up the interpolator for given grids.
   * \grids Array containing the grids corresponding to the first degree
   * dimensions of the tensor to interpolate.
   */
  RegularGridInterpolator(std::array<Vector, degree> grids) : grids_(grids) {}

  /** Compute interpolation weights and indices for interpolation points.
   * @param positions Eigen matrix containing the positions at which to
   * interpolate the given tensor.
   */
  InterpolationWeights
  calculate_weights(const eigen::MatrixFixedRows<Scalar, degree>& positions) const {
    eigen::MatrixFixedRows<Scalar, degree> weights(positions.rows(), degree);
    eigen::MatrixFixedRows<Eigen::Index, degree> indices(positions.rows(),
                                                         degree);
    for (size_t i = 0; i < degree; ++i) {
      detail::calculate_weights(weights.col(i),
                                indices.col(i),
                                grids_[i],
                                positions.col(i));
    }
    return std::make_pair(weights, indices);
  }


  /** Interpolate tensor using precomputed weights.
   * @param t The tensor to interpolate.
   * @param interp_weights The interpolation weights precomuted using the
   * calculate weights member function.
   * @param positions Eigen matrix containing the positions at which to
   * interpolate t.
   */
  auto interpolate(const Tensor& t,
                   const InterpolationWeights& interp_weights) const
      -> std::vector<
          typename detail::InterpolationResult<Tensor, degree>::type> {
    const WeightMatrix& weights =
        std::get<0>(interp_weights);
    const IndexMatrix& indices =
        std::get<1>(interp_weights);
    using ResultType =
        typename detail::InterpolationResult<Tensor, degree>::type;
    std::vector<ResultType> results;

    int n_results = weights.rows();
    results.resize(n_results);
    for (int i = 0; i < n_results; ++i) {
        results[i] = scattering::interpolate<Tensor, degree, Scalar>(t,
                                                        weights.row(i),
                                                        indices.row(i));
    }
    return results;
  }

  /** Interpolate tensor using precomputed weights.
   * @param t The tensor to interpolate.
   * @param interp_weights The interpolation weights precomuted using the
   * calculate weights member function.
   * @param positions Eigen matrix containing the positions at which to
   * interpolate t.
   */
  // pxx :: hide
  template <typename ResultContainer>
  void interpolate(ResultContainer results,
                   const Tensor& t,
                   const InterpolationWeights& interp_weights) const {
    const WeightMatrix& weights = std::get<0>(interp_weights);
    const IndexMatrix& indices = std::get<1>(interp_weights);

    int n_results = weights.rows();
    for (int i = 0; i < n_results; ++i) {
        eigen::tensor_index<1>(results, {i}) = scattering::interpolate<Tensor, degree, Scalar>(t,
                                                                                    weights.row(i),
                                                                                    indices.row(i));
    }
  }

  /** Interpolate tensor at given positions.
   * @param t The tensor to interpolate.
   * @param positions Eigen matrix containing the positions at which to
   * interpolate t.
   */
  auto interpolate(const Tensor& t,
                   eigen::MatrixFixedRows<Scalar, degree> positions) const
      -> std::vector<
      typename detail::InterpolationResult<Tensor, degree>::type> {
      auto interp_weights = calculate_weights(positions);
      return interpolate(t, interp_weights);
  }

 protected:
  std::array<Vector, degree> grids_;
};

template <int ... Axis>
struct RegridImpl;

template <int i, int a, int... Axis>
struct RegridImpl<i, a, Axis...> {
  template <typename Tensor, typename Weights, typename Indices>
      __attribute__((always_inline)) auto static inline compute(const Tensor& t,
                             std::array<Eigen::DenseIndex, Tensor::NumIndices> coords,
                             const Weights& weights,
                             const Indices& indices) {
    auto w = weights[i][coords[a]];
    auto index = indices[i][coords[a]];
    decltype(coords) new_coords = coords;

    new_coords[a] = index;
    auto l = RegridImpl<i + 1, Axis...>::compute(t, new_coords, weights, indices);
    if (w == 1.0) {
      return l;
    }
    new_coords[a] += 1;
    auto r = RegridImpl<i + 1, Axis...>::compute(t, new_coords, weights, indices);
    return w * l + r * (static_cast<decltype(w)>(1.0) - w);
  }
};

template <int i>
struct RegridImpl<i> {
  template <typename Tensor, typename Weights, typename Indices>
      __attribute__((always_inline)) auto static inline compute(const Tensor& t,
                             std::array<Eigen::DenseIndex, Tensor::NumIndices> coords,
                             const Weights&/*weights*/,
                             const Indices&/*indices*/) {
    return t(coords);
  }
};


/** Regridder for regular grids.
 *
 * The RegularRegridder implements regridding of regular grids. It interpolates
 * a gridded tensor to new grids along a given subset of its dimensions.
 *
 * @tparam Scalar The type used to represent scalars in the tensor.
 * @tparam rank The rank of the tensor to regrid.
 * @tparam n_dimensions The number of dimensions to regrid.
 */
template <typename Scalar, int ... Axes>
class RegularRegridder {
 public:

    static constexpr  eigen::Index n_dimensions = sizeof...(Axes);

  /** Dimensions of output tensor.
   * @param in The tensor to regrid
   * @returns std::array containing the dimensions of the regridded tensor.
   */
  // pxx :: hide
  template <typename Tensor,
            typename IndexArray =
                std::array<typename Tensor::Index, Tensor::NumIndices>>
  IndexArray get_output_dimensions(const Tensor& in) {
    auto input_dimensions = in.dimensions();
    IndexArray output_dimensions;
    std::copy(input_dimensions.begin(),
              input_dimensions.end(),
              output_dimensions.begin());
    for (eigen::Index i = 0; i < n_dimensions; ++i) {
      output_dimensions[dimensions_[i]] = new_grids_[i].size();
    }
    return output_dimensions;
  }

  /** Get strides of output tensor.
   * @param in The tensor to regrid
   * @returns std::array containing the strides of the regridded tensor.
   */
  // pxx :: hide
  template <typename Tensor,
            typename IndexArray =
                std::array<typename Tensor::Index, Tensor::NumIndices>>
  IndexArray get_strides(const Tensor& t) {
    auto dimensions = get_output_dimensions(t);
    IndexArray strides;
    eigen::Index c = 1;
    for (eigen::Index i = Tensor::NumIndices - 1; i >= 0; --i) {
      strides[i] = c;
      c *= dimensions[i];
    }
    return strides;
  }

  /** Convert single-integer index to tensor-index array.
   * @param index The single-integer index
   * @strides index The strides of the tensor
   * @returns std::array containing the tensor-index array corresponding to
   * index.
   */
  // pxx :: hide
  template <typename Index, typename IndexArray>
  static inline IndexArray get_indices(Index index, IndexArray strides) {
    IndexArray indices{0};
    for (eigen::Index i = 0; i < strides.size(); ++i) {
      if (i > 0) {
        indices[i] = (index % strides[i - 1]) / strides[i];
      } else {
        indices[i] = index / strides[i];
      }
    }
    return indices;
  }

  /** Sets up the regridder for given grids.
   * @param old_grids std::vector containing the old grids, which should be
   * regridded.
   * @param new_grids std::vector containing the new grids
   * @param dimensions Vector containing the tensor dimensions to which the
   * given grids correspond.
   */
  RegularRegridder(std::array<eigen::Vector<Scalar>, n_dimensions> old_grids,
                   std::array<eigen::Vector<Scalar>, n_dimensions> new_grids)
      : old_grids_(old_grids), new_grids_(new_grids) {
    for (size_t i = 0; i < dimensions_.size(); ++i) {
      auto ws = detail::calculate_weights<Scalar>(old_grids_[i], new_grids_[i]);
      weights_[i] = std::get<0>(ws);
      indices_[i] = std::get<1>(ws);
    }
  }

  /** Regrid tensor.
   * @param in The tensor to regrid.
   * @return The regridded tensor.
   */
  template <typename Tensor>
  Tensor regrid(const Tensor &input) {
    constexpr int rank = Tensor::NumIndices;
    eigen::Tensor<typename Tensor::Scalar, rank> output{get_output_dimensions(input)};

    using IndexArray = std::array<Eigen::DenseIndex, rank>;
    auto generator = [this, &input](const IndexArray &coordinates) {
        return RegridImpl<0, Axes ...>::compute(input,
                                                coordinates,
                                                this->weights_,
                                                this->indices_);
    };

    return output.generate(generator);
  }

  /** Regrid tensor.
   * @param output The tensor to hold the result.
   * @param input The tensor to regrid.
   * @return The regridded tensor.
   */
  // pxx :: hide
  template <typename TensorOut, typename TensorIn>
    void regrid(TensorOut &output, TensorIn &input) {
    constexpr int rank = TensorOut::NumIndices;

    using IndexArray = std::array<Eigen::DenseIndex, rank>;
    auto generator = [this, &input](const IndexArray &coordinates) {
        return RegridImpl<0, Axes ...>::compute(input,
                                                coordinates,
                                                this->weights_,
                                                this->indices_);
    };

    output = output.generate(generator);
  }

 protected:


  std::array<eigen::Vector<Scalar>, n_dimensions> old_grids_;
  std::array<eigen::Vector<Scalar>, n_dimensions> new_grids_;
  std::array<eigen::Vector<Scalar>, n_dimensions> weights_;
  std::array<eigen::Vector<eigen::Index>, n_dimensions> indices_;
  static constexpr std::array<int, n_dimensions> dimensions_{Axes ...};

};

// pxx :: export
// pxx :: instance(["1", "Eigen::Tensor<float, 2, Eigen::RowMajor>", "float"])
// pxx :: instance(["1", "Eigen::Tensor<double, 2, Eigen::RowMajor>", "double"])
template <eigen::Index rank, typename TensorType, typename Scalar>
eigen::Tensor<typename TensorType::Scalar, TensorType::NumIndices> downsample_dimension(
    const TensorType& input,
    const eigen::Vector<Scalar>& input_grid,
    const eigen::Vector<Scalar>& output_grid,
    Scalar minimum_value,
    Scalar maximum_value) {
  using eigen::Index;
  using eigen::Tensor;
  using eigen::Vector;

  // Calculate integration limits.
  Index n = output_grid.size();
  Vector<Scalar> limits(n + 1);
  limits[0] = minimum_value;
  limits[n] = maximum_value;
  for (Index i = 1; i < n; ++i) {
    limits[i] = 0.5 * (output_grid[i - 1] + output_grid[i]);
  }

  // Interpolate input to boundary values.
  auto regridder = RegularRegridder<Scalar, rank>({input_grid}, {limits});
  auto regridded = regridder.regrid(input);

  // Prepare output
  auto result_dimensions = input.dimensions();
  result_dimensions[rank] = output_grid.size();
  auto result = Tensor<typename TensorType::Scalar, TensorType::NumIndices>{result_dimensions};

  // Prepare integral value
  std::array<Index, TensorType::NumIndices - 1> integral_dimensions;
  Index dimension_index = 0;
  for (Index i = 0; i < TensorType::NumIndices; ++i) {
    if (i != rank) {
      integral_dimensions[dimension_index] = input.dimension(i);
      dimension_index += 1;
    }
  }
  auto integral =
      Tensor<typename TensorType::Scalar, TensorType::NumIndices - 1>(integral_dimensions);

  Index input_index = 0;

  for (Index i = 0; i < n; ++i) {
    Scalar left_limit = limits[i];
    Scalar right_limit = limits[i + 1];

    Scalar left = left_limit;
    Scalar right = left_limit;
    Scalar dx = 0.0;

    auto left_value = regridded.template chip<rank>(i);
    auto right_value = left_value;

    integral = integral.setZero();

    while ((input_index < input_grid.size()) && (input_grid[input_index] < right_limit)) {
      right_value = input.template chip<rank>(input_index);
      right = input_grid[input_index];
      integral += 0.5 * (right - left) * (left_value + right_value);
      dx += (right - left);
      left = right;
      left_value = right_value;
      input_index += 1;
    }

    right_value = regridded.template chip<rank>(i + 1);
    right = right_limit;
    dx += (right - left);
    integral += 0.5 * (right - left) * (left_value + right_value);
    if (i == 0) {
      if (n == 1) {
        dx = limits[1] - limits[0];
      } else {
        dx = output_grid[0] - limits[0] +
             0.5 * (output_grid[1] - output_grid[0]);
      }
    } else if (i < n - 1) {
      dx = 0.5 * (output_grid[i + 1] - output_grid[i - 1]);
    } else {
      dx = limits[i + 1] - output_grid[i] +
           0.5 * (output_grid[i] - output_grid[i - 1]);
    }
    if (dx != 0.0) {
      result.template chip<rank>(i) = 1.0 / dx * integral;
    } else {
      result.template chip<rank>(i) = integral.setZero();
    }
  }

  return result;
}

}  // Namespace scattering

#endif
