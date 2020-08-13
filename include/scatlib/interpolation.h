/** \file interpolation.h
 *
 * Generic interpolation method for Eigen tensors on regular
 * grids.
 *
 * @author Simon Pfreundschuh, 2020
 */
#ifndef __SCATLIB_INTERPOLATION__
#define __SCATLIB_INTERPOLATION__

#include <scatlib/eigen.h>

#include <algorithm>
#include <utility>
#include <chrono>
#include <iostream>
#include <type_traits>
#include <vector>

namespace scatlib {
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

  static inline Result compute(
      const Tensor& tensor,
      const eigen::VectorFixedSize<Scalar, N> &weights,
      const eigen::VectorFixedSize<Eigen::Index, N> &indices,
      const eigen::VectorFixedSize<Eigen::Index, I> &offsets = {}) {
    eigen::VectorFixedSize<Eigen::Index, N> indices_new{indices};
    for (Eigen::Index i = 0; i < I; ++i) {
      indices_new[i] += offsets[i];
    }

    eigen::VectorFixedSize<Eigen::Index, I + 1> offsets_new{};
    for (Eigen::Index i = 0; i < I; ++i) {
      offsets_new[i] = offsets[i];
    }
    offsets_new[I] = 0;

    Scalar w = weights[I];
    Result t = Interpolator<Tensor, N, I + 1>::compute(tensor,
                                                       weights,
                                                       indices,
                                                       offsets_new);
    if (w < 1.0) {
      offsets_new[I] = 1;
      t = w * t;
      t += static_cast<Scalar>(1.0 - w) *
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

  static inline Result compute(
      const Tensor& tensor,
      const eigen::VectorFixedSize<Scalar, N> &/*weights*/,
      const eigen::VectorFixedSize<Eigen::Index, N> &indices,
      const eigen::VectorFixedSize<Eigen::Index, N> &offsets = {}) {
    std::array<Eigen::Index, N> indices_new;
    for (Eigen::Index i = 0; i < N; ++i) {
      indices_new[i] = indices[i] + offsets[i];
    }
    return eigen::tensor_index(tensor, indices_new);
  }
};


template <typename Derived>
    struct Interpolator<Eigen::Map<Derived>, 2, 2> {
    using Matrix = Eigen::Map<Derived>;
    using Scalar = typename Derived::Scalar;

    static inline Scalar compute(
        const Matrix& matrix,
        const eigen::VectorFixedSize<Scalar, 2> &/*weights*/,
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

    template <typename T>
    static inline auto compute(
        const Eigen::Map<T>& matrix,
        const eigen::VectorFixedSize<Scalar, 1> &/*weights*/,
        const eigen::VectorFixedSize<Eigen::Index, 1> &indices,
        const eigen::VectorFixedSize<Eigen::Index, 1> &offsets = {},
        typename std::enable_if<!is_vector<T>::value>::type * = 0) {
        return matrix.row(indices[0] + offsets[0]);
    }

    template <typename T>
    static inline Scalar compute(
        const Eigen::Map<T>& vector,
        const eigen::VectorFixedSize<Scalar, 1> &/*weights*/,
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
                       const PositionVector& positions) {
  using Scalar = typename std::remove_reference<WeightVector>::type::Scalar;

  for (int i = 0; i < positions.size(); ++i) {
    auto p = positions[i];
    auto f = std::lower_bound(grid.begin(), grid.end(), p);
    if (f != grid.end()) {
      if (f == grid.begin()) {
        indices[i] = 0;
        weights[i] = 1.0;
      } else {
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
      }
    } else {
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
// pxx :: instance(["Eigen::Tensor<float, 4, Eigen::RowMajor>", "3"])
// pxx :: instance(["Eigen::Tensor<float, 5, Eigen::RowMajor>", "3"])
// pxx :: instance(["Eigen::Tensor<float, 6, Eigen::RowMajor>", "3"])
// pxx :: instance(["Eigen::Tensor<float, 7, Eigen::RowMajor>", "3"])
// pxx :: instance(["Eigen::Tensor<float, 3, Eigen::RowMajor>", "2"])
// pxx :: instance(["Eigen::Tensor<float, 4, Eigen::RowMajor>", "2"])
// pxx :: instance(["Eigen::Tensor<float, 5, Eigen::RowMajor>", "2"])
// pxx :: instance(["Eigen::Tensor<float, 6, Eigen::RowMajor>", "2"])
// pxx :: instance(["Eigen::Tensor<float, 7, Eigen::RowMajor>", "2"])
// pxx :: instance(["Eigen::Tensor<float, 2, Eigen::RowMajor>", "1"])
// pxx :: instance(["Eigen::Tensor<float, 3, Eigen::RowMajor>", "1"])
// pxx :: instance(["Eigen::Tensor<float, 4, Eigen::RowMajor>", "1"])
// pxx :: instance(["Eigen::Tensor<float, 5, Eigen::RowMajor>", "1"])
// pxx :: instance(["Eigen::Tensor<float, 6, Eigen::RowMajor>", "1"])
// pxx :: instance(["Eigen::Tensor<float, 7, Eigen::RowMajor>", "1"])
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
template <typename Tensor, size_t degree>
inline typename detail::InterpolationResult<Tensor, degree>::type interpolate(
    const Tensor& tensor,
    Eigen::Ref<const eigen::VectorFixedSize<typename Tensor::Scalar, degree>>
        weights,
    Eigen::Ref<const eigen::VectorFixedSize<typename Tensor::Index, degree>>
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
  using Scalar = typename Tensor::Scalar;
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
      results[i] = scatlib::interpolate<Tensor, degree>(t,
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
  template <typename ResultContainer>
  void interpolate(ResultContainer results,
                   const Tensor& t,
                   const InterpolationWeights& interp_weights) const {
    const WeightMatrix& weights = std::get<0>(interp_weights);
    const IndexMatrix& indices = std::get<1>(interp_weights);

    int n_results = weights.rows();
    results.resize(n_results);
    for (int i = 0; i < n_results; ++i) {
      results[i] = scatlib::interpolate<Tensor, degree>(t,
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

// pxx :: export
// pxx :: instance(["double", "4", "2"])
/** Regridder for regular grids.
 *
 * The RegularRegridder implements regridding of regular grids. It interpolates
 * a gridded tensor to new grids along a given subset of its dimensions.
 *
 * @tparam Scalar The type used to represent scalars in the tensor.
 * @tparam rank The rank of the tensor to regrid.
 * @tparam n_dimensions The number of dimensions to regrid.
 */
template <typename Scalar, eigen::Index rank, eigen::Index n_dimensions>
class RegularRegridder {
 public:
  /** Dimensions of output tensor.
   * @param in The tensor to regrid
   * @returns std::array containing the dimensions of the regridded tensor.
   */
  std::array<eigen::Index, rank> get_output_dimensions(
      eigen::Tensor<Scalar, rank>& in) {
    auto input_dimensions = in.dimensions();
    std::array<eigen::Index, rank> output_dimensions;
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
  std::array<eigen::Index, rank> get_strides(eigen::Tensor<Scalar, rank>& t) {
    auto dimensions = get_output_dimensions(t);
    std::array<eigen::Index, rank> strides;
    eigen::Index c = 1;
    for (eigen::Index i = rank - 1; i >= 0; --i) {
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
  static std::array<eigen::Index, rank> get_indices(
      eigen::Index index,
      std::array<eigen::Index, rank> strides) {
    std::array<eigen::Index, rank> indices{0};
    for (eigen::Index i = 0; i < rank; ++i) {
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
                   std::array<eigen::Vector<Scalar>, n_dimensions> new_grids,
                   std::array<eigen::Index, n_dimensions> dimensions)
      : old_grids_(old_grids), new_grids_(new_grids), dimensions_(dimensions) {
    for (size_t i = 0; i < dimensions.size(); ++i) {
      auto ws = detail::calculate_weights<Scalar>(old_grids_[i], new_grids_[i]);
      weights_[i] = std::get<0>(ws);
      indices_[i] = std::get<1>(ws);
    }
  }

  /** Regrid tensor.
   * @param in The tensor to regrid.
   * @return The regridded tensor.
   */
  eigen::Tensor<Scalar, rank> regrid(eigen::Tensor<Scalar, rank> in) {
    using WeightVector = eigen::VectorFixedSize<Scalar, rank>;
    using IndexVector = eigen::VectorFixedSize<eigen::Index, rank>;
    using Tensor = eigen::Tensor<Scalar, rank>;
    WeightVector interpolation_weights = WeightVector::Constant(1.0);
    IndexVector interpolation_indices = IndexVector::Constant(0);
    eigen::Tensor<Scalar, rank> output{get_output_dimensions(in)};
    std::array<eigen::Index, rank> strides = get_strides(output);

    for (eigen::Index i = 0; i < output.size(); ++i) {
      auto output_indices = get_indices(i, strides);
      for (eigen::Index j = 0; j < rank; ++j) {
        interpolation_indices[j] = output_indices[j];
      }
      for (eigen::Index j = 0; j < n_dimensions; ++j) {
        eigen::Index dim = dimensions_[j];
        interpolation_weights[dim] = weights_[j][output_indices[dim]];
        interpolation_indices[dim] = indices_[j][output_indices[dim]];
      }
      output.coeffRef(output_indices) =
          interpolate<Tensor, rank>(in,
                                    interpolation_weights,
                                    interpolation_indices);
    }
    return output;
  }

 protected:
  std::array<eigen::Vector<Scalar>, n_dimensions> old_grids_;
  std::array<eigen::Vector<Scalar>, n_dimensions> new_grids_;
  std::array<eigen::Index, n_dimensions> dimensions_;
  std::array<eigen::Vector<Scalar>, n_dimensions> weights_;
  std::array<eigen::Vector<eigen::Index>, n_dimensions> indices_;
};

}  // namespace scatlib

#endif
