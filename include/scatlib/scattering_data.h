/** \file scattering_data.h
 *
 * Provides the ScatteringData class providing methods for the manipulation
 * of scattering data.
 *
 * @author Simon Pfreundschuh, 2020
 */
#ifndef __SCATLIB_SCATTERING_DATA__
#define __SCATLIB_SCATTERING_DATA__

#include <scatlib/eigen.h>
#include <scatlib/sht.h>
#include <scatlib/interpolation.h>
#include <memory>
#include <cassert>

namespace scatlib {

template <typename Scalar, int rank>
using SpectralTensorType = eigen::Tensor<std::complex<Scalar>, rank>;
template <typename Scalar, int rank>
using GriddedTensorType = eigen::Tensor<Scalar, rank>;

namespace detail {

/** Transform fully-gridded to spectral representation.
 * Class to transform fully-gridded scattering data to spectral
 * format along the scattering angles.
 */
template <typename Scalar>
class GriddedToSpectralTransformer {
public:
  /** Initialize transformer.
   *
   * @param t The input tensor to transform.
   */
  GriddedToSpectralTransformer(scatlib::sht::SHT &sht)
      : sht_(sht),
        n_lat_(sht.get_n_latitudes()),
        n_lon_(sht.get_n_longitudes()),
        nlm_scat_(sht.get_number_of_spectral_coeffs()) {}

  /** Get major stride for input.
   *
   * The major stride of the input tensor is the distance between different
   * scattering-angle sequences.
   * @return the major input stride of the input data.
   */
  size_t get_major_stride_in() { return n_lat_ * n_lon_; }

  /** Get major stride for output.
   *
   * The major stride of the output tensor is the distance between different
   * sequences of spherical harmonics coefficients.
   * @return the major input stride of the input data.
   */
  size_t get_major_stride_out() { return nlm_scat_; }

  /// Map pointing to the two-dimensions coefficient field for given scattering
  /// angles.
  template <int rank>
  eigen::ConstMatrixMap<Scalar> get_input_sequence(
      size_t major_index,
      const GriddedTensorType<Scalar, rank> &in) {
    size_t start = major_index * get_major_stride_in();
    return eigen::ConstMatrixMap<Scalar>(in.data(), n_lon_, n_lat_);
  }

  /// Map pointing to spherical harmonics coefficient vector in input tensor.
  template <int rank>
  eigen::VectorMap<std::complex<Scalar>> get_output_sequence(
      size_t major_index,
      SpectralTensorType<Scalar, rank> &out) {
    size_t start = major_index * get_major_stride_out();
    return eigen::VectorMap<std::complex<Scalar>>(out.data(), nlm_scat_);
  }

  /// Dimensions of the output (transformed) tensor.
  template <int rank>
    std::array<eigen::Index, rank - 1ul> get_output_dimensions(
      const GriddedTensorType<Scalar, rank> &in) {
    auto input_dimensions = in.dimensions();
    std::array<eigen::Index, rank - 1ul> output_dimensions;
    std::copy(input_dimensions.begin(),
              input_dimensions.end() - 1,
              output_dimensions.begin());
    output_dimensions[rank - 1] = nlm_scat_;
    return output_dimensions;
  }

  template <int rank>
      SpectralTensorType<Scalar, rank - 1> transform(
      const GriddedTensorType<Scalar, rank> &in) {
      SpectralTensorType<Scalar, rank - 1> out{get_output_dimensions(in)};
    for (size_t i = 0; i < out.size() / get_major_stride_out(); ++i) {
        eigen::Matrix<Scalar> spatial_coeffs = get_input_sequence(i, in);
        get_output_sequence(i, out) = sht_.transform(spatial_coeffs);
    }
    return out;
  }

 protected:
  scatlib::sht::SHT &sht_;
  size_t n_lat_, n_lon_, nlm_scat_;
};

/** Transform spectral to fully-spectral.
 *
 * Class to transform spectral scattering data to fully-spectral
 * format.
 *
 */
template <typename Scalar>
class SpectralToFullySpectralTransformer {
public:
  SpectralToFullySpectralTransformer(size_t nlm_scat, scatlib::sht::SHT &sht)
      : sht_(sht),
        n_lat_(sht.get_n_latitudes()),
        n_lon_(sht.get_n_longitudes()),
        nlm_scat_(nlm_scat),
        nlm_inc_(sht.get_number_of_spectral_coeffs()) {}

  /**
   * The stride between sequences of SH coefficient over incoming angles
   * in the input tensor.
   */
  size_t get_major_stride_in() { return n_lat_ * n_lon_ * nlm_scat_ * 2; }

  /**
   * The stride between incoming SH coefficient-sequences in the output tensor.
   */
  size_t get_major_stride_out() { return 2 * nlm_scat_ * nlm_inc_; }

  /**
   * The stride between sequences of scattering SH-coefficients corresponding to
   * different l- and m-indices in the input tensor.
   */
  size_t get_lm_stride_in() { return nlm_scat_ * 2; }

  /**
   * The stride between sequences of incoming SH-coefficients corresponding to
   * different l- and m-indices in the input tensor.
   */
  size_t get_lm_stride_out() { return nlm_scat_; }

  /**
   * Matrix map pointing to sequence of scattering SH coefficients in the input
   * tensor.
   */
  template <int rank>
  eigen::MatrixMap<Scalar> get_input_sequence(
      size_t major_index,
      size_t lm_index,
      size_t complex_index,
      SpectralTensorType<Scalar, rank> &in) {
    size_t start = major_index * get_major_stride_in() +
                   lm_index * get_lm_stride_in() + complex_index;
    size_t col_stride = 2 * get_lm_stride_in();
    size_t row_stride = 2 * get_lm_stride_in() * n_lat_;
    return eigen::MatrixMap<Scalar>(in.data(),
                                    n_lon_,
                                    n_lat_,
                                    {row_stride, col_stride});
  }

  /**
   * Vector map pointing to sequences of incoming SH coefficients in the input
   * tensor.
   */
  template <int rank>
  eigen::VectorMap<std::complex<Scalar>> get_output_sequence(
      size_t major_index,
      size_t lm_index,
      size_t complex_index,
      SpectralTensorType<Scalar, rank> &out) {
    size_t start = major_index * get_major_stride_out() +
                   complex_index * get_major_stride_out() / 2 +
                   lm_index * get_lm_stride_out();
    size_t stride = get_lm_stride_out();
    return eigen::VectorMap<Scalar>(out.data(), nlm_inc_, {stride});
  }

  /// Dimensions of the output (transformed) tensor.
  template <int rank>
      std::array<eigen::Index, rank> get_output_dimensions(
      SpectralTensorType<Scalar, rank> &in) {
    auto input_dimensions = in.dimensions();
    std::array<size_t, rank> output_dimensions;
    std::copy(input_dimensions.begin(),
              input_dimensions.end(),
              output_dimensions.begin());
    output_dimensions[rank - 3] = nlm_inc_;
    output_dimensions[rank - 4] = 2;
    return output_dimensions;
  }

  template <int rank>
  SpectralTensorType<Scalar, rank> transform(
      SpectralTensorType<Scalar, rank> &in) {
    SpectralTensorType<Scalar, rank> out(get_output_dimensions(in));
    for (size_t i = 0; i < out.size() / get_major_stride_out(); ++i) {
      for (size_t j = 0; j < nlm_scat_; ++j) {
        get_output_sequence(i, j, 0, out) = get_input_sequence(i, j, 0);
        get_output_sequence(i, j, 1, out) = get_input_sequence(i, j, 1);
      }
    }
    return out;
  }

 protected:
  scatlib::sht::SHT &sht_;
  size_t n_lat_, n_lon_, nlm_scat_, nlm_inc_;
};
}  // namespace detail

//
// Scattering data.
//

enum class DataFormat { Gridded, Spectral, FullySpectral};
enum class DataType {Spherical, TotallyRandom, AzimuthallyRandom, General};

/** Base class for scattering data.
 * Holds frequency and temperature grids.
 */
class ScatteringDataBase {
 public:
  static DataType determine_type(size_t n_azimuth_angles_inc,
                                 size_t n_zenith_angles_inc,
                                 size_t n_azimuth_angles_scat,
                                 size_t n_zenith_angles_scat) {
    if ((n_zenith_angles_scat == 1) && (n_azimuth_angles_scat == 1) &&
        (n_zenith_angles_inc == 1) && (n_azimuth_angles_inc == 1)) {
      return DataType::Spherical;
    }
    if ((n_azimuth_angles_scat == 1) && (n_zenith_angles_inc == 1) &&
        (n_azimuth_angles_inc == 1)) {
      return DataType::TotallyRandom;
    }
    if (n_azimuth_angles_inc == 1) {
      return DataType::TotallyRandom;
    }
    return DataType::General;
  }

  ScatteringDataBase(size_t n_azimuth_angles_inc,
                     size_t n_zenith_angles_inc,
                     size_t n_azimuth_angles_scat,
                     size_t n_zenith_angles_scat)
      : type_(determine_type(n_azimuth_angles_inc,
                            n_zenith_angles_inc,
                            n_azimuth_angles_scat,
                            n_zenith_angles_scat)) {}

  DataType get_type() const { return type_; }

 protected:
  DataType type_;
};

template <typename Scalar, DataFormat format>
class ScatteringData;

// pxx :: export
// pxx :: instance("ScatteringDataGridded", ["double"])
template <typename Scalar>
class ScatteringData<Scalar, DataFormat::Gridded> : public ScatteringDataBase {
 public:

  using CoeffType = Scalar;
  using VectorType = eigen::Vector<Scalar>;
  using VectorMapType = eigen::VectorMap<Scalar>;
  using MatrixType = eigen::Matrix<Scalar>;
  template <int rank>
  using TensorType = eigen::Tensor<Scalar, rank>;
  template <int rank>
  using TensorMapType = eigen::TensorMap<Scalar, rank>;
  template <int rank>
  using ConstTensorMapType = eigen::ConstTensorMap<Scalar, rank>;

  using ScatteringDataBase::get_type;

  template <typename Tensor, int N>
      MatrixType interpolate_angles(const eigen::MatrixFixedRows<Scalar, N> &angles,
                                    std::array<VectorMapType, N> angle_grids,
                                    const Tensor &data) const {
      eigen::Index n_points = angles.rows();
      eigen::Index n_cols_output = data.dimension(0);
      eigen::Matrix<Scalar> result(n_points, n_cols_output);
      using Interpolator = RegularGridInterpolator<typename eigen::Map<const Tensor>::type,
                                                 N,
                                                 VectorMapType>;
    auto interpolator = Interpolator(angle_grids);
    auto weights = interpolator.calculate_weights(angles);

    std::array<Eigen::Index, N> index{0};
    for (eigen::Index i = 0; i < n_cols_output; ++i) {
      index[0] = i;
      interpolator.interpolate(result.col(i), data(index), weights);
    }
    return result;
  }

  template <typename Tensor, int N>
  VectorType interpolate_angles_scalar(
      const eigen::MatrixFixedRows<Scalar, N> &angles,
      std::array<VectorMapType, N> angle_grids,
      const Tensor &data) const {
    eigen::Index n_points = angles.rows();
    VectorType result(n_points);
    using Interpolator =
        RegularGridInterpolator<typename eigen::Map<const Tensor>::type,
                                N,
                                VectorMapType>;
    auto interpolator = Interpolator(angle_grids);
    auto weights = interpolator.calculate_weights(angles);
    interpolator.interpolate(result, data, weights);
    return result;
  }

  ScatteringData(VectorType azimuth_angles_inc,
                 VectorType zenith_angles_inc,
                 VectorType azimuth_angles_scat,
                 VectorType zenith_angles_scat,
                 TensorType<5> phase_matrix,
                 TensorType<3> extinction_matrix,
                 TensorType<3> absorption_vector,
                 TensorType<2> backscattering_coeff,
                 TensorType<2> forwardscattering_coeff)
      : n_azimuth_angles_inc_(azimuth_angles_inc.size()),
        n_zenith_angles_inc_(zenith_angles_inc.size()),
        n_azimuth_angles_scat_(azimuth_angles_scat.size()),
        n_zenith_angles_scat_(zenith_angles_scat.size()),
        ScatteringDataBase(azimuth_angles_inc.size(),
                           zenith_angles_inc.size(),
                           azimuth_angles_scat.size(),
                           zenith_angles_scat.size()),
        azimuth_angles_inc_(azimuth_angles_inc),
        zenith_angles_inc_(zenith_angles_inc),
        azimuth_angles_scat_(azimuth_angles_scat),
        zenith_angles_scat_(zenith_angles_scat),
        azimuth_angle_map_inc_(azimuth_angles_inc_.data(),
                               n_azimuth_angles_inc_),
        zenith_angle_map_inc_(zenith_angles_inc_.data(), n_zenith_angles_inc_),
        azimuth_angle_map_scat_(azimuth_angles_scat_.data(),
                                n_azimuth_angles_scat_),
        zenith_angle_map_scat_(zenith_angles_scat_.data(),
                               n_zenith_angles_scat_),
        phase_matrix_(phase_matrix),
        extinction_matrix_(extinction_matrix),
        absorption_vector_(absorption_vector),
        backscattering_coeff_(backscattering_coeff),
        forwardscattering_coeff_(forwardscattering_coeff) {}

  VectorType &get_azimuth_angles_inc() { return azimuth_angles_inc_; }
  const eigen::Vector<Scalar> & get_azimuth_angles_inc() const {return azimuth_angles_inc_;}
  VectorType & get_zenith_angles_inc() {return zenith_angles_inc_;}
  const eigen::Vector<Scalar> & get_zenith_angles_inc() const {return zenith_angles_inc_;}
  VectorType & get_azimuth_angles_scattering() {return azimuth_angles_scat_;}
  const eigen::Vector<Scalar> & get_azimuth_angles_scattering() const {return azimuth_angles_scat_;}
  VectorType & get_zenith_angles_scattering() {return zenith_angles_scat_;}
  const eigen::Vector<Scalar> & get_zenith_angles_scattering() const {return zenith_angles_scat_;}

  size_t get_n_azimuth_angles_inc() const { return n_azimuth_angles_inc_; }
  size_t get_n_zenith_angles_inc() const { return n_zenith_angles_inc_; }
  size_t get_n_azimuth_angles_scat() const { return n_azimuth_angles_scat_; }
  size_t get_n_zenith_angles_scat() const { return n_zenith_angles_scat_; }

  //
  // Phase matrix.
  //

  const eigen::Tensor<Scalar, 5> &get_phase_matrix_data() const { return phase_matrix_; }
  eigen::Tensor<Scalar, 5> &get_phase_matrix_data() { return phase_matrix_; };

  Scalar get_phase_matrix() const {
      assert(get_type() == DataType::Spherical);
      return phase_matrix_({0, 0, 0, 0, 0});
  }

  MatrixType get_phase_matrix(
      const eigen::MatrixFixedRows<Scalar, 1> &angles) const {

    eigen::Index phase_matrix_size = phase_matrix_.dimension(0);
    eigen::Index n_points = angles.rows();

    auto type = get_type();
    if (type == DataType::Spherical) {
        return MatrixType::Constant(n_points, phase_matrix_size, get_phase_matrix());
    }
    assert(get_type() == DataType::TotallyRandom);

    eigen::Matrix<Scalar> result(n_points, phase_matrix_size);
    using Interpolator = RegularGridInterpolator<eigen::ConstVectorMap<Scalar>,
                                                 1,
                                                 VectorMapType>;
    auto interpolator = Interpolator({zenith_angle_map_scat_});
    auto weights = interpolator.calculate_weights(angles);

    for (eigen::Index i = 0; i < phase_matrix_size; ++i) {
      std::array<Eigen::Index, 4> index{i, 0, 0, 0};
      interpolator.interpolate(result.col(i), phase_matrix_(index), weights);
    }
    return result;
  }

  const eigen::Matrix<Scalar> get_phase_matrix(
      const eigen::MatrixFixedRows<Scalar, 3> &angles) const {

      eigen::Index phase_matrix_size = phase_matrix_.dimension(0);
      eigen::Index n_points = angles.rows();

      auto type = get_type();
      if (type == DataType::Spherical) {
          return MatrixType::Constant(n_points, phase_matrix_size, get_phase_matrix());
      }
      if (type == DataType::TotallyRandom) {
          return get_phase_matrix(static_cast<eigen::MatrixFixedRows<Scalar, 1>>(angles.col(2)));
      }
      assert(get_type() == DataType::AzimuthallyRandom);

      eigen::Matrix<Scalar> result(n_points, phase_matrix_size);
      using Interpolator = RegularGridInterpolator<ConstTensorMapType<3>, 3, VectorMapType>;
      auto interpolator = Interpolator({azimuth_angle_map_inc_,
                                        zenith_angle_map_inc_,
                                        zenith_angle_map_scat_});
      auto weights = interpolator.calculate_weights(angles);

      for (eigen::Index i = 0; i < phase_matrix_size; ++i) {
          std::array<Eigen::Index, 2> index{i, 0};
          interpolator.interpolate(result.col(i), phase_matrix_(index), weights);
      }
      return result;
  }

  const eigen::Matrix<Scalar> get_phase_matrix(
      const eigen::MatrixFixedRows<Scalar, 4> &angles) const {

      eigen::Index phase_matrix_size = phase_matrix_.dimension(0);
      eigen::Index n_points = angles.rows();

      auto type = get_type();
      if (type == DataType::Spherical) {
          return MatrixType::Constant(n_points, phase_matrix_size, get_phase_matrix());
      }
      if (type == DataType::TotallyRandom) {
          return get_phase_matrix(static_cast<eigen::MatrixFixedRows<Scalar, 1>>(angles.col(3)));
      }
      if (type == DataType::AzimuthallyRandom) {
          return get_phase_matrix(static_cast<eigen::MatrixFixedRows<Scalar, 3>>(angles.rightCols(1)));
      }
      assert(get_type() == DataType::AzimuthallyRandom);

      eigen::Matrix<Scalar> result(n_points, phase_matrix_size);
      using Interpolator = RegularGridInterpolator<ConstTensorMapType<4>, 4, VectorMapType>;
      auto interpolator = Interpolator({zenith_angle_map_inc_,
                                        azimuth_angle_map_inc_,
                                        zenith_angle_map_inc_,
                                        zenith_angle_map_scat_});
      auto weights = interpolator.calculate_weights(angles);

      for (eigen::Index i = 0; i < phase_matrix_size; ++i) {
          std::array<Eigen::Index, 1> index{i};
          interpolator.interpolate(result.col(i), phase_matrix_(index), weights);
      }
      return result;
  }

  //
  // Extinction matrix interpolation.
  //

  const TensorType<3> &get_extinction_matrix_data() const {
      return extinction_matrix_;
  };
  TensorType<3> &get_extinction_matrix_data() { return extinction_matrix_; };

  Scalar get_extinction_matrix() const {
      auto type = get_type();
      assert((type == DataType::Spherical) || (type == DataType::TotallyRandom));
      return extinction_matrix_(0, 0, 0);
  };

  MatrixType get_extinction_matrix(
      const eigen::MatrixFixedRows<Scalar, 1> &angles) const {
    eigen::Index extinction_matrix_size = extinction_matrix_.dimension(0);
    eigen::Index n_points = angles.rows();

    auto type = get_type();
    if ((type == DataType::Spherical || type == DataType::TotallyRandom)) {
      return MatrixType::Constant(n_points,
                                  extinction_matrix_size,
                                  get_extinction_matrix());
    }
    assert(get_type() == DataType::AzimuthallyRandom);

    eigen::Matrix<Scalar> result(n_points, extinction_matrix_size);
    using Interpolator = RegularGridInterpolator<eigen::ConstVectorMap<Scalar>,
                                                 1,
                                                 VectorMapType>;
    auto interpolator = Interpolator({zenith_angle_map_inc_});
    auto weights = interpolator.calculate_weights(angles);

    for (eigen::Index i = 0; i < extinction_matrix_size; ++i) {
      std::array<Eigen::Index, 2> index{i, 0};
      interpolator.interpolate(result.col(i),
                               extinction_matrix_(index),
                               weights);
    }
    return result;
  }

  MatrixType get_extinction_matrix(
      const eigen::MatrixFixedRows<Scalar, 2> &angles) const {
    eigen::Index extinction_matrix_size = extinction_matrix_.dimension(0);
    eigen::Index n_points = angles.rows();

    auto type = get_type();
    if ((type == DataType::Spherical || type == DataType::TotallyRandom)) {
      return MatrixType::Constant(n_points,
                                  extinction_matrix_size,
                                  get_extinction_matrix());
    }
    if (type == DataType::AzimuthallyRandom) {
        return get_extinction_matrix(static_cast<eigen::MatrixFixedRows<Scalar, 1>>(angles.col(1)));
    }
    assert(get_type() == DataType::AzimuthallyRandom);

    eigen::Matrix<Scalar> result(n_points, extinction_matrix_size);
    using Interpolator = RegularGridInterpolator<eigen::ConstMatrixMap<Scalar>,
                                                 2,
                                                 VectorMapType>;
    auto interpolator = Interpolator({azimuth_angle_map_inc_, zenith_angle_map_inc_});
    auto weights = interpolator.calculate_weights(angles);

    for (eigen::Index i = 0; i < extinction_matrix_size; ++i) {
      std::array<Eigen::Index, 1> index{i};
      interpolator.interpolate(result.col(i),
                               extinction_matrix_(index),
                               weights);
    }
    return result;
  }

  //
  // Absorption vector interpolation.
  //

  const TensorType<3> &get_absorption_vector_data() const {
      return absorption_vector_;
  };
  TensorType<3> &get_absorption_vector_data() { return absorption_vector_; };

  Scalar get_absorption_vector() const {
      auto type = get_type();
      assert((type == DataType::Spherical) || (type == DataType::TotallyRandom));
      return absorption_vector_(0, 0, 0);
  };

  MatrixType get_absorption_vector(
      const eigen::MatrixFixedRows<Scalar, 1> &angles) const {
    eigen::Index absorption_vector_size = absorption_vector_.dimension(0);
    eigen::Index n_points = angles.rows();

    auto type = get_type();
    if ((type == DataType::Spherical || type == DataType::TotallyRandom)) {
      return MatrixType::Constant(n_points,
                                  absorption_vector_size,
                                  get_absorption_vector());
    }
    assert(get_type() == DataType::AzimuthallyRandom);

    eigen::Matrix<Scalar> result(n_points, absorption_vector_size);
    using Interpolator = RegularGridInterpolator<eigen::ConstVectorMap<Scalar>,
                                                 1,
                                                 VectorMapType>;
    auto interpolator = Interpolator({zenith_angle_map_inc_});
    auto weights = interpolator.calculate_weights(angles);

    for (eigen::Index i = 0; i < absorption_vector_size; ++i) {
      std::array<Eigen::Index, 2> index{i, 0};
      interpolator.interpolate(result.col(i),
                               absorption_vector_(index),
                               weights);
    }
    return result;
  }

  MatrixType get_absorption_vector(
      const eigen::MatrixFixedRows<Scalar, 2> &angles) const {
    eigen::Index absorption_vector_size = absorption_vector_.dimension(0);
    eigen::Index n_points = angles.rows();

    auto type = get_type();
    if ((type == DataType::Spherical || type == DataType::TotallyRandom)) {
      return MatrixType::Constant(n_points,
                                  absorption_vector_size,
                                  get_absorption_vector());
    }
    if (type == DataType::AzimuthallyRandom) {
        return get_absorption_vector(static_cast<eigen::MatrixFixedRows<Scalar, 1>>(angles.col(1)));
    }
    assert(get_type() == DataType::AzimuthallyRandom);

    eigen::Matrix<Scalar> result(n_points, absorption_vector_size);
    using Interpolator = RegularGridInterpolator<eigen::ConstMatrixMap<Scalar>,
                                                 2,
                                                 VectorMapType>;
    auto interpolator = Interpolator({azimuth_angle_map_inc_, zenith_angle_map_inc_});
    auto weights = interpolator.calculate_weights(angles);

    for (eigen::Index i = 0; i < absorption_vector_size; ++i) {
      std::array<Eigen::Index, 1> index{i};
      interpolator.interpolate(result.col(i),
                               absorption_vector_(index),
                               weights);
    }
    return result;
  }

  //
  // Backscattering coefficient.
  //

  const TensorType<2> &get_backscattering_coeff_data() const {
      return backscattering_coeff_;
  };
  TensorType<2> &get_backscattering_coeff_data() {
      return backscattering_coeff_;
  }

  Scalar get_backscattering_coeff() const {
      return backscattering_coeff_({0, 0});
  }

  VectorType get_backscattering_coeff(
      const eigen::MatrixFixedRows<Scalar, 1> &angles) const {
    eigen::Index n_points = angles.rows();

    auto type = get_type();
    if ((type == DataType::Spherical || type == DataType::TotallyRandom)) {
      return VectorType::Constant(n_points, get_backscattering_coeff());
    }
    assert(get_type() == DataType::AzimuthallyRandom);

    std::array<eigen::Index, 1> index = {0};
    return interpolate_angles_scalar(angles, {zenith_angle_map_inc_}, backscattering_coeff_(index));
  }

  VectorType get_backscattering_coeff(
      const eigen::MatrixFixedRows<Scalar, 2> &angles) const {
    eigen::Index n_points = angles.rows();

    auto type = get_type();
    if ((type == DataType::Spherical || type == DataType::TotallyRandom)) {
      return VectorType::Constant(n_points, get_backscattering_coeff());
    }
    if (type == DataType::AzimuthallyRandom) {
      return get_backscattering_coeff(static_cast<eigen::MatrixFixedRows<Scalar, 1>>(angles.col(1)));
    }
    assert(get_type() == DataType::General);

    auto tensor_map = ConstTensorMapType<2>(backscattering_coeff_.data(), backscattering_coeff_.dimensions());
    return interpolate_angles_scalar(angles,
                                     {azimuth_angle_map_inc_, zenith_angle_map_inc_},
                                     tensor_map);
  }

  //
  // Forwardscattering coefficient.
  //

  const TensorType<2> &get_forwardscattering_coeff_data() const {
      return forwardscattering_coeff_;
  };
  TensorType<2> &get_forwardscattering_coeff_data() {
      return forwardscattering_coeff_;
  }

  Scalar get_forwardscattering_coeff() const {
      return forwardscattering_coeff_({0, 0});
  }

  VectorType get_forwardscattering_coeff(
      const eigen::MatrixFixedRows<Scalar, 1> &angles) const {
    eigen::Index n_points = angles.rows();

    auto type = get_type();
    if ((type == DataType::Spherical || type == DataType::TotallyRandom)) {
      return VectorType::Constant(n_points, get_forwardscattering_coeff());
    }
    assert(get_type() == DataType::AzimuthallyRandom);

    std::array<eigen::Index, 1> index = {0};
    return interpolate_angles_scalar(angles, {zenith_angle_map_inc_}, forwardscattering_coeff_(index));
  }

  VectorType get_forwardscattering_coeff(
      const eigen::MatrixFixedRows<Scalar, 2> &angles) const {
    eigen::Index n_points = angles.rows();

    auto type = get_type();
    if ((type == DataType::Spherical || type == DataType::TotallyRandom)) {
      return VectorType::Constant(n_points, get_forwardscattering_coeff());
    }
    if (type == DataType::AzimuthallyRandom) {
      return get_forwardscattering_coeff(static_cast<eigen::MatrixFixedRows<Scalar, 1>>(angles.col(1)));
    }
    assert(get_type() == DataType::General);

    auto tensor_map = ConstTensorMapType<2>(forwardscattering_coeff_.data(), forwardscattering_coeff_.dimensions());
    return interpolate_angles_scalar(angles,
                                     {azimuth_angle_map_inc_, zenith_angle_map_inc_},
                                     tensor_map);
  }

 protected:
  int n_azimuth_angles_inc_;
  int n_zenith_angles_inc_;
  int n_azimuth_angles_scat_;
  int n_zenith_angles_scat_;

  VectorType azimuth_angles_inc_;
  VectorType zenith_angles_inc_;
  VectorType azimuth_angles_scat_;
  VectorType zenith_angles_scat_;
  VectorMapType azimuth_angle_map_inc_;
  VectorMapType zenith_angle_map_inc_;
  VectorMapType azimuth_angle_map_scat_;
  VectorMapType zenith_angle_map_scat_;

  TensorType<5> phase_matrix_;       // elements x (inc. ang.) x (scat. ang.)
  TensorType<3> extinction_matrix_;  // elements x (inc. ang.) x (scat. ang.)
  TensorType<3> absorption_vector_;  // elements x (inc. ang.) x (scat. ang.)
  TensorType<2> backscattering_coeff_;  // (inc. ang.)
  TensorType<2> forwardscattering_coeff_;  // (inc. angles)

  std::shared_ptr<sht::SHT> sht_;

};

// pxx :: export
// pxx :: instance("ScatteringDataSpectral", ["double"])
template <typename Scalar>
  class ScatteringData<Scalar, DataFormat::Spectral> : public ScatteringDataBase {
 public:
  using CoeffType = std::complex<Scalar>;
  using VectorType = eigen::Vector<Scalar>;
  using VectorMapType = eigen::VectorMap<Scalar>;
  using CmplxVectorType = eigen::Vector<std::complex<Scalar>>;
  using MatrixType = eigen::Matrix<Scalar>;
  using CmplxMatrixType = eigen::Matrix<std::complex<Scalar>>;
  template <int rank>
      using TensorType = eigen::Tensor<Scalar, rank>;
  template <int rank>
      using TensorMapType = eigen::TensorMap<Scalar, rank>;
  template <int rank>
      using ConstTensorMapType = eigen::ConstTensorMap<Scalar, rank>;
  template <int rank>
      using CmplxTensorType = eigen::Tensor<std::complex<Scalar>, rank>;

  using ScatteringDataBase::get_type;

  ScatteringData(VectorType azimuth_angles_inc,
                 VectorType zenith_angles_inc,
                 CmplxTensorType<4> phase_matrix,
                 TensorType<3> extinction_matrix,
                 TensorType<3> absorption_vector,
                 TensorType<2> backscattering_coeff,
                 TensorType<2> forwardscattering_coeff,
                 std::shared_ptr<sht::SHT> sht_scat)
      : n_azimuth_angles_inc_(azimuth_angles_inc.size()),
        n_zenith_angles_inc_(zenith_angles_inc.size()),
        n_azimuth_angles_scat_(sht_scat->get_n_longitudes()),
        n_zenith_angles_scat_(sht_scat->get_n_latitudes()),
        ScatteringDataBase(n_azimuth_angles_inc_,
                           n_zenith_angles_inc_,
                           n_azimuth_angles_scat_,
                           n_zenith_angles_scat_),
        azimuth_angles_inc_(azimuth_angles_inc),
        zenith_angles_inc_(zenith_angles_inc),
        azimuth_angle_map_inc_(azimuth_angles_inc_.data(),
                               n_azimuth_angles_inc_),
        zenith_angle_map_inc_(zenith_angles_inc_.data(), n_zenith_angles_inc_),
        phase_matrix_(phase_matrix),
        extinction_matrix_(extinction_matrix),
        absorption_vector_(absorption_vector),
        backscattering_coeff_(backscattering_coeff),
      forwardscattering_coeff_(forwardscattering_coeff),
      sht_scat_(sht_scat) {}

  VectorType & get_azimuth_angles_inc() {return azimuth_angles_inc_;}
  const eigen::Vector<Scalar> & get_azimuth_angles_inc() const {return azimuth_angles_inc_;}
  VectorType & get_zenith_angles_inc() {return zenith_angles_inc_;}
  const eigen::Vector<Scalar> & get_zenith_angles_inc() const {return zenith_angles_inc_;}

  size_t get_n_azimuth_angles_inc() const { return azimuth_angles_inc_.size(); }
  size_t get_n_zenith_angles_inc() const { return zenith_angles_inc_.size(); }

  //
  // Phase matrix.
  //

  const CmplxTensorType<4> &get_phase_matrix_data() const { return phase_matrix_; }
  CmplxTensorType<4> &get_phase_matrix_data() { return phase_matrix_; };

  std::complex<Scalar> get_phase_matrix() const {
      assert(get_type() == DataType::Spherical);
      return phase_matrix_({0, 0, 0, 0});
  }

  MatrixType get_phase_matrix(
      const eigen::Vector<Scalar> &angles) const {

      int n_points = angles.rows();
      int n_outputs = phase_matrix_.dimension(0);
      MatrixType result(n_points, n_outputs);

      for (int i = 0; i < n_outputs; ++i) {
          std::array<eigen::Index, 3> index = {i, 0, 0};
          result.col(i) = sht_scat_->evaluate(static_cast<CmplxVectorType>(phase_matrix_(index)),
                                              angles);
      }

      return result;
  }

  const TensorType<3> &get_extinction_matrix() const {
      return extinction_matrix_;
  };
  TensorType<3> &get_extinction_matrix() { return extinction_matrix_; };
  const TensorType<3> &get_absorption_vector() const {
      return absorption_vector_;
  };
  TensorType<3> &get_absorption_vector() { return absorption_vector_; };
  const TensorType<2> &get_backscattering_coeff() const {
      return backscattering_coeff_;
  };
  TensorType<2> &get_backscattering_coeff() {
      return backscattering_coeff_;
  }
  const TensorType<2> &get_forwardscattering_coeff() const {
      return forwardscattering_coeff_;
  }
  TensorType<2> &get_forwardscattering_coeff() {
      return forwardscattering_coeff_;
  }

 protected:

  int n_azimuth_angles_inc_;
  int n_zenith_angles_inc_;
  int n_azimuth_angles_scat_;
  int n_zenith_angles_scat_;

  VectorType azimuth_angles_inc_;
  VectorType zenith_angles_inc_;
  VectorMapType azimuth_angle_map_inc_;
  VectorMapType zenith_angle_map_inc_;

  CmplxTensorType<4> phase_matrix_;                   // elements x (inc. ang.) x nlm
  TensorType<3> extinction_matrix_;              // elements x (inc. ang.) x nlm
  TensorType<3> absorption_vector_;              // elements x (inc. ang.) x nlm
  TensorType<2> backscattering_coeff_;     // (inc. ang.) x nlm
  TensorType<2> forwardscattering_coeff_;  // (inc. ang.) x nlm

  std::shared_ptr<sht::SHT> sht_scat_;
};

// pxx :: export
// pxx :: instance("ScatteringDataFullySpectral", ["double"])
template <typename Scalar>
class ScatteringData<Scalar, DataFormat::FullySpectral> {
 public:
  using CoeffType = std::complex<Scalar>;
  using VectorType = eigen::Vector<Scalar>;
  template <int rank>
  using TensorType = eigen::Tensor<CoeffType, rank>;

 public:
  ScatteringData(
                 TensorType<4> phase_matrix,
                 TensorType<4> extinction_matrix,
                 TensorType<4> absorption_vector,
                 TensorType<3> backscattering_coeff,
                 TensorType<3> forwardscattering_coeff)
      : phase_matrix_(phase_matrix),
        extinction_matrix_(extinction_matrix),
        absorption_vector_(absorption_vector),
        backscattering_coeff_(backscattering_coeff),
        forwardscattering_coeff_(forwardscattering_coeff) {}

  const TensorType<4> &get_phase_matrix() const { return phase_matrix_; }
  TensorType<4> &get_phase_matrix() { return phase_matrix_; };
  const TensorType<4> &get_extinction_matrix() const {
    return extinction_matrix_;
  };
  TensorType<4> &get_extinction_matrix() { return extinction_matrix_; };
  const TensorType<4> &get_absorption_vector() const {
    return absorption_vector_;
  };
  TensorType<4> &get_absorption_vector() { return absorption_vector_; };
  const TensorType<3> &get_backscattering_coeff() const {
    return backscattering_coeff_;
  };
  TensorType<3> &get_backscattering_coeff() {
    return backscattering_coeff_;
  }
  const TensorType<3> &get_forwardscattering_coeff() const {
    return forwardscattering_coeff_;
  }
  TensorType<3> &get_forwardscattering_coeff() {
    return forwardscattering_coeff_;
  }

 protected:
  TensorType<4> phase_matrix_;                   // elements x nlm_inc x nlm x 2
  TensorType<4> extinction_matrix_;              // elements x nlm_inc x nlm x 2
  TensorType<4> absorption_vector_;              // elements x nlm_inc x nlm x 2
  TensorType<3> backscattering_coeff_;     // nlm_inc x nlm x 2
  TensorType<3> forwardscattering_coeff_;  // nlm_inc x nlm x 2
};

// pxx :: export
// pxx :: instance(["double"])
template <typename Scalar>
ScatteringData<Scalar, DataFormat::Spectral> gridded_to_spectral(
    const ScatteringData<Scalar, DataFormat::Gridded> &in,
    size_t l_max,
    size_t m_max) {
    size_t n_lat = in.get_n_zenith_angles_scat(); 
    n_lat = n_lat - n_lat % 2;
  size_t n_lon = in.get_n_azimuth_angles_scat();

  std::shared_ptr<sht::SHT> sht_scat = std::make_shared<sht::SHT>(l_max, m_max, n_lat, n_lon);

  auto transformer = detail::GriddedToSpectralTransformer<Scalar>(*sht_scat);

  auto zenith_angles_inc = in.get_zenith_angles_inc();
  auto azimuth_angles_inc = in.get_azimuth_angles_inc();
  auto phase_matrix = transformer.transform(in.get_phase_matrix_data());
  auto extinction_matrix = in.get_extinction_matrix_data();
  auto absorption_vector = in.get_absorption_vector_data();
  auto backscattering_coeff = in.get_backscattering_coeff_data();
  auto forwardscattering_coeff = in.get_forwardscattering_coeff_data();

  auto spectral = ScatteringData<Scalar, DataFormat::Spectral>(zenith_angles_inc,
                                                               azimuth_angles_inc,
                                                               phase_matrix,
                                                               extinction_matrix,
                                                               absorption_vector,
                                                               backscattering_coeff,
                                                               forwardscattering_coeff,
                                                               sht_scat);
  return spectral;
}

}  // namespace scatlib

#endif
