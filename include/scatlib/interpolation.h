/** \file interpolation.h
 *
 * Generic interpolation method for Eigen tensors on regular
 * grids.
 *
 * @author Simon Pfreundschuh, 2020
 */
#ifndef __SCATLIB_INTERPOLATION__
#define __SCATLIB_INTERPOLATION__

#include <algorithm>
#include <iostream>
#include <scatlib/eigen.h>


namespace scatlib {
namespace detail {

//
// Interpolation helper.
//

template <typename Tensor, Eigen::Index N>
struct InterpolationResult {
  using TensorType = Eigen::Tensor<typename Tensor::Scalar,
                                   Tensor::NumIndices - N, Tensor::Options>;
  using MatrixType =
      Eigen::Matrix<typename Tensor::Scalar, -1, -1, Tensor::Options>;
  using VectorType =
      Eigen::Matrix<typename Tensor::Scalar, 1, -1, Tensor::Options>;
  using ScalarType = typename Tensor::Scalar;

  template <typename T,
            typename std::enable_if<(T::NumIndices > N + 2)>::type* = nullptr>
  static TensorType test_fun(T* ptr);
  template <typename T,
            typename std::enable_if<(T::NumIndices == N + 2)>::type* = nullptr>
  static MatrixType test_fun(T* ptr);
  template <typename T,
            typename std::enable_if<(T::NumIndices == N + 1)>::type* = nullptr>
  static VectorType test_fun(T* ptr);
  template <typename T,
            typename std::enable_if<(T::NumIndices == N)>::type* = nullptr>
  static ScalarType test_fun(T* ptr);

  using type = decltype(test_fun(reinterpret_cast<Tensor*>(0)));
};

template <typename Tensor, Eigen::Index N, Eigen::Index I = 0>
struct Interpolator {
  using Result = typename InterpolationResult<Tensor, N>::type;
  using Scalar = typename Tensor::Scalar;

  static inline Result compute(const Tensor& tensor,
                               eigen::VectorFixedSize<Scalar, N> weights,
                               eigen::VectorFixedSize<Eigen::Index, N> indices,
                               eigen::VectorFixedSize<Eigen::Index, I> offsets = {}) {
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
    Result t = Interpolator<Tensor, N, I + 1>::compute(tensor, weights, indices,
                                                       offsets_new);
    if (w < 1.0) {
      offsets_new[I] = 1;
      t = w * t;
      t += static_cast<Scalar>(1.0 - w) *
           Interpolator<Tensor, N, I + 1>::compute(tensor, weights, indices,
                                                   offsets_new);
    }
    return t;
  }
};

template <typename Tensor, Eigen::Index N>
struct Interpolator<Tensor, N, N> {
  using Result = typename InterpolationResult<Tensor, N>::type;
  using Scalar = typename Tensor::Scalar;

  static inline Result compute(const Tensor& tensor,
                               eigen::VectorFixedSize<Scalar, N> /*weights*/,
                               eigen::VectorFixedSize<Eigen::Index, N> indices,
                               eigen::VectorFixedSize<Eigen::Index, N> offsets = {}) {
    std::array<Eigen::Index, N> indices_new;
    for (Eigen::Index i = 0; i < N; ++i) {
      indices_new[i] = indices[i] + offsets[i];
    }
    return tensor(indices_new);
  }
};

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
    using WeightIndexPair = std::pair<eigen::Vector<Scalar>,
    eigen::Vector<Eigen::Index>>;

template <typename Scalar>
WeightIndexPair<Scalar> calculate_weights(
    const eigen::Vector<Scalar>& grid,
    const eigen::Vector<Scalar>& positions) {
  if (positions.size() == 0) {
    return std::make_pair<eigen::Vector<Scalar>, eigen::Vector<Eigen::Index>>({},
                                                                          {});
  }
  auto indices_sorted = indirect_sort(positions);

  int index = 0;
  int position_index = indices_sorted[index];
  auto p = positions[position_index];

  eigen::Vector<Scalar> weights = eigen::Vector<Scalar>::Zero(positions.size());
  eigen::Vector<Eigen::Index> indices =
      eigen::Vector<Eigen::Index>::Zero(positions.size());

  for (int i = 0; i < grid.size() - 1; ++i) {
    auto& left = grid[i];
    auto& right = grid[i + 1];
    auto df = 1.0 / (right - left);
    while ((p < right) && (index < positions.size())) {
      if (p < left) {
        weights[position_index] = 1.0;
      } else {
        weights[position_index] = df * (right - p);
      }
      indices[position_index] = i;
      index += 1;
      if (index < positions.size()) {
        position_index = indices_sorted[index];
        p = positions[position_index];
      }
    }
  }

  for (int i = index; i < positions.size(); ++i) {
    weights[indices_sorted[i]] = 1.0;
    indices[indices_sorted[i]] = grid.size() - 1;
  }

  return std::make_pair(weights, indices);
}

}  // namespace detail

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
 * Piece-wise linear interpolation of the given tensor along its first dimensions.
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
    const Tensor &tensor,
    Eigen::Ref<const eigen::VectorFixedSize<typename Tensor::Scalar, degree>> weights,
    Eigen::Ref<const eigen::VectorFixedSize<typename Tensor::Index, degree>> indices) {
  return detail::Interpolator<Tensor, degree>::compute(tensor, weights, indices);
}

// pxx :: export
// pxx :: instance(["Eigen::Tensor<double, 5, Eigen::RowMajor>", "3"])
//
/** Regular grid interpolator.
 *
 * Piecewise-linear interpolator on regular grids.
 *
 * @tparam Tensor The Eigen tensor type to interpolate.
 * @tparam degree Along how many dimensions to interpolate.
 */
template <typename Tensor, size_t degree>
class RegularGridInterpolator {
public:
 using Scalar = typename Tensor::Scalar;
 using WeightVector = eigen::VectorFixedSize<Scalar, degree>;
 using IndexVector = eigen::VectorFixedSize<Eigen::Index, degree>;

 /** Sets up the interpolator for given grids.
  * \grids Array containing the grids corresponding to the first degree
  * dimensions of the tensor to interpolate.
  */
 RegularGridInterpolator(std::array<eigen::Vector<Scalar>, degree> grids)
     : grids_(grids) {}

 /** Compute interpolation weights and indices for interpolation points.
  * @param positions Eigen matrix containing the positions at which to
  * interpolate the given tensor.
  */
 std::pair<eigen::MatrixFixedRows<Scalar, degree>, eigen::MatrixFixedRows<typename Eigen::Index, degree>>
 get_weights(eigen::MatrixFixedRows<Scalar, degree> positions) const {
   eigen::MatrixFixedRows<Scalar, degree> weights(positions.rows(), degree);
   eigen::MatrixFixedRows<Eigen::Index, degree> indices(positions.rows(), degree);
   for (size_t i = 0; i < degree; ++i) {
     auto ws = detail::calculate_weights<Scalar>(grids_[i], positions.col(i));
     weights.col(i) = std::get<0>(ws);
     indices.col(i) = std::get<1>(ws);
   }
   return std::make_pair(weights, indices);
 }

 /** Interpolate tensor at given positions.
  * @param t The tensor to interpolate.
  * @param positions Eigen matrix containing the positions at which to interpolate t.
  */
 auto interpolate(const Tensor& t, eigen::MatrixFixedRows<Scalar, degree> positions) const
     -> std::vector<
         typename detail::InterpolationResult<Tensor, degree>::type> {
   eigen::MatrixFixedRows<Scalar, degree> weights;
   eigen::MatrixFixedRows<Eigen::Index, degree> indices;
   std::tie(weights, indices) = get_weights(positions);
   using ResultType =
       typename detail::InterpolationResult<Tensor, degree>::type;
   std::vector<ResultType> results;
   for (int i = 0; i < positions.rows(); ++i) {
       results.push_back(scatlib::interpolate<Tensor, degree>(t, weights.row(i), indices.row(i)));
   }
   return results;
 }

protected:
 std::array<eigen::Vector<Scalar>, degree> grids_;
};

// pxx :: export
// pxx :: instance(["double", "4", "2"])
template <typename Scalar, eigen::Index rank, eigen::Index n_dimensions>
class RegularRegridder {
 public:
  std::array<eigen::Index, rank> get_output_dimensions(
      eigen::Tensor<Scalar, rank>& in) {
    auto input_dimensions = in.dimensions();
    std::array<eigen::Index, rank> output_dimensions;
    std::copy(input_dimensions.begin(), input_dimensions.end(),
              output_dimensions.begin());
    for (eigen::Index i = 0; i < n_dimensions; ++i) {
      output_dimensions[dimensions_[i]] = new_grids_[i].size();
    }
    return output_dimensions;
  }

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

  static std::array<eigen::Index, rank> get_indices(
      eigen::Index index, std::array<eigen::Index, rank> strides) {
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

 public:
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
      output.coeffRef(output_indices) = interpolate<Tensor, rank>(in, interpolation_weights, interpolation_indices);
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
