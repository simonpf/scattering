#ifndef __SCATLIB_INTERPOLATION__
#define __SCATLIB_INTERPOLATION__

#include <Eigen/Core>
#include <Eigen/CXX11/Tensor>

template<typename Scalar>
using EigenMatrix = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;
template<typename Scalar>
using EigenVector = Eigen::Matrix<Scalar, -1, 1, Eigen::RowMajor>;

namespace scatlib {
namespace detail {


template <typename Tensor, Eigen::Index N>

struct InterpolationResult {
  using TensorType = Eigen::Tensor<typename Tensor::Scalar, Tensor::NumIndices - N,
                             Tensor::Options>;
  using MatrixType = Eigen::Matrix<typename Tensor::Scalar, -1, -1, Tensor::Options>;
  using VectorType = Eigen::Matrix<typename Tensor::Scalar, 1, -1, Tensor::Options>;
  using ScalarType = typename Tensor::Scalar;

    template <typename T, typename std::enable_if<(T::NumIndices > N + 2)>::type* = nullptr >
  static TensorType test_fun(T* ptr);
    template <typename T, typename std::enable_if<(T::NumIndices == N + 2)>::type* = nullptr >
  static MatrixType test_fun(T* ptr);
    template <typename T, typename std::enable_if<(T::NumIndices == N + 1)>::type* = nullptr >
  static VectorType test_fun(T* ptr);
    template <typename T, typename std::enable_if<(T::NumIndices == N)>::type* = nullptr>
   static ScalarType test_fun(T* ptr);

    using type = decltype(test_fun(reinterpret_cast<Tensor *>(0)));
};


    template<typename Tensor, Eigen::Index N, Eigen::Index I = 0>
    struct Interpolator {

        using Result = typename InterpolationResult<Tensor, N>::type;
        using Scalar = typename Tensor::Scalar;

        static inline Result compute(const Tensor &tensor,
                                     std::array<Scalar, N> weights,
                                     std::array<Eigen::Index, N> indices,
                                     std::array<Eigen::Index, I> offsets = {}) {
            std::array<Eigen::Index, N> indices_new{indices};
            for (Eigen::Index i = 0; i < I; ++i) {
                indices_new[i] += offsets[i];
            }

            std::array<Eigen::Index, I + 1> offsets_new{};
            for (Eigen::Index i = 0; i < I; ++i) {
                offsets_new[i] = offsets[i];
            }
            offsets_new[I] = 0;

            Scalar w = weights[I];
            Result t = Interpolator<Tensor, N, I + 1>::compute(tensor, weights, indices, offsets_new);
            if (w < 1.0) {
                offsets_new[I] = 1;
                t = w * t;
                t += static_cast<Scalar>(1.0 - w) * Interpolator<Tensor, N, I + 1>::compute(tensor, weights, indices, offsets_new);
            }
            return t;
        }
    };

    template<typename Tensor, Eigen::Index N>
    struct Interpolator<Tensor, N, N> {
        using Result = typename InterpolationResult<Tensor, N>::type;
        using Scalar = typename Tensor::Scalar;

        static inline Result compute(const Tensor &tensor,
                                     std::array<Scalar, N> /*weights*/,
                                     std::array<Eigen::Index, N> indices,
                                     std::array<Eigen::Index, N> offsets = {}) {
            std::array<Eigen::Index, N> indices_new{indices};
            for (Eigen::Index i = 0; i < N; ++i) {
                indices_new[i] += offsets[i];
            }
            return tensor(indices_new);
        }
    };
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
////////////////////////////////////////////////////////////////////////////////
// Interpolate
////////////////////////////////////////////////////////////////////////////////
/** Interpolate tensor from weights.
 *
 *
 */
template <typename Tensor, size_t N>
typename detail::InterpolationResult<Tensor, N>::type interpolate(const Tensor tensor,
                 std::array<typename Tensor::Scalar, N> weights,
                 std::array<typename Tensor::Index, N> indices) {
    return detail::Interpolator<Tensor, N>::compute(tensor, weights, indices);
}

template <typename Scalar>
using WeightIndexPair = std::pair<EigenVector<Scalar>, EigenVector<Eigen::Index>>;

template <typename Scalar>
WeightIndexPair<Scalar> calulate_weights(const EigenVector<Scalar> &grid,
                                         const EigenVector<Scalar> &positions) {
    // Check that positions are sorted.
    if (!is_sorted(positions.begin(), positions.end())) {
        std::sort(positions.begin(), positions.end());
    }

    if (positions.size()) {
        return 0;
    }

    size_t position_index = 0;
    auto p = positions[0];

    EigenVector<Scalar> weights = positions.zero();
    EigenVector<Eigen::Index> indices(positions.size());

    for (int i = 0; i < grid.size() - 1; ++i) {
        auto &g = grid[i];
        if (i == 0) {
            while (p < g) {
                weights[position_index] = 1.0;
                indices[position_index] = 0;
                position_index += 1;
                p = positions[position_index];
            }
        } else {
            auto df = 1.0 / (grid[i + 1] - g);
            while (p < g) {
                weights[position_index] = df * (p - g);
                indices[position_index] = i;
                position_index += 1;
                p = positions[position_index];
            }
        }
    }

    for (int i = position_index; i < positions.size(); ++i) {
        weights[i] = 1.0;
        indices[i] = grid.size() - 1;
    }

    return std::make_pair(weights, indices);
}


// pxx :: export
// pxx :: instance(["Eigen::Tensor<float, 6, Eigen::RowMajor>", "3"])
template <typename Tensor, size_t degree>
struct RegularGridInterpolator {
  using Scalar = typename Tensor::Scalar;

RegularGridInterpolator(std::array <EigenVector<Scalar>, degree> grids)
      : grids_(grids) {}

  std::pair<EigenMatrix<Scalar>, EigenMatrix<Eigen::Index>> get_weights(
      EigenMatrix<Scalar> positions) const {
    EigenMatrix<Scalar> weights(positions.size(), degree);
    EigenMatrix<Eigen::Index> indices(positions.size(), degree);
    for (size_t i = 0; i < degree; ++i) {
        std::tie(weights.col(i), indices.col(i)) = calculate_weights(grids_[i], positions.col(i));
    }
    return std::make_pair(weights, indices);
  }

  auto interpolate(const Tensor& t, EigenMatrix<Scalar> positions) const
      -> std::vector<typename detail::InterpolationResult<Tensor, degree>::type> {
    EigenMatrix<Scalar> weights;
    EigenMatrix<Eigen::Index> indices;
    std::tie(weights, indices) = get_weights(positions);
    using ResultType = typename detail::InterpolationResult<Tensor, degree>::type;
    std::vector<ResultType> results;
    for (size_t i = 0; i < positions.rows(); ++i) {
      results.push_back(interpolate(t, weights.row(i), indices.row(i)));
    }
    return results;
  }

 protected:
  std::array<EigenVector<Scalar>, degree> grids_;
};
}

#endif
