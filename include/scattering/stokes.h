/** \file stokes.h
 *
 * Transformation and expansion of scattering data.
 *
 * @author Simon Pfreundschuh, 2020
 */
#ifndef __SCATTERING_STOKES__
#define __SCATTERING_STOKES__

#include "scattering/utils/math.h"
#include "scattering/eigen.h"
#include "scattering/scattering_data_field.h"
#include "scattering/utils/math.h"

namespace scattering {
namespace stokes {

/// [1, 1] element of phase matrix stored in compact format.
template <typename VectorType>
auto f11(const VectorType &v) {
  return v[0];
}

/// [1, 2] element of phase matrix stored in compact format.
template <typename VectorType>
auto f12(const VectorType &v) {
  return v[1];
}

/// [2, 2] element of phase matrix stored in compact format.
template <typename VectorType>
auto f22(const VectorType &v) {
  return v[2];
}

/// [3, 3] element of phase matrix stored in compact format.
template <typename VectorType>
auto f33(const VectorType &v) {
  return v[3];
}

/// [3, 4] element of phase matrix stored in compact format.
template <typename VectorType>
auto f34(const VectorType &v) {
  return v[4];
}

/// [4, 4] element of phase matrix stored in compact format.
template <typename VectorType>
auto f44(const VectorType &v) {
  return v[5];
}

////////////////////////////////////////////////////////////////////////////////
// Calculation of rotation coefficients.
////////////////////////////////////////////////////////////////////////////////

/** Calculate scattering angle from incoming and outgoing directions.
 * @param lon_inc The incoming-angle longitude component.
 * @param lat_inc The incoming-angle latitude component.
 * @param lon_scat The outgoing (scattering) angle longitude component.
 * @param lat_scat The outgoing (scattering) angle longitude component..
 * @return The angle between the incoming and outgoing directions.
 */
template <typename Scalar>
Scalar scattering_angle(Scalar lon_inc,
                        Scalar lat_inc,
                        Scalar lon_scat,
                        Scalar lat_scat) {
    Scalar cos_theta = cos(lat_inc) * cos(lat_scat) +
        sin(lat_inc) * sin(lat_scat) * cos(lon_scat - lon_inc);
    return math::save_acos(cos_theta);
}

/** Calculates the rotation coefficients for scattering matrix.
 *
 * This method calculates the rotation coefficients that are required to
 * transform a phase function of a randomly-oriented particle to the scattering
 * matrix, which describes its scattering behavior w.r.t. to the reference
 * frame. This equation calculates the angle Theta and the coefficients C_1,
 * C_2, S_1, S_2 as defined in equation (4.16) of
 * "Scattering, Absorption, and Emission of Light by Small Particles."
 *
 * @param lon_inc The longitude component of the incoming angle.
 * @param lat_inc The latitude component of the incoming angle.
 * @param lon_scat The longitude component of the scattering angle.
 * @param lat_scat The latitude component of the scattering angle.
 *
 */
template <typename Scalar>
std::array<Scalar, 5> rotation_coefficients(Scalar lon_inc,
                                            Scalar lat_inc,
                                            Scalar lon_scat,
                                            Scalar lat_scat) {
  Scalar cos_theta = cos(lat_inc) * cos(lat_scat) +
                     sin(lat_inc) * sin(lat_scat) * cos(lon_scat - lon_inc);
  Scalar theta = math::save_acos(cos_theta);
  if ((math::small(abs(lon_scat - lon_inc))) ||
      (math::equal(abs(lon_scat - lon_inc), 2.0 * M_PI))) {
    theta = abs(lat_inc - lat_scat);
  } else if ((math::equal(lon_scat - lon_inc, 2.0 * M_PI))) {
    theta = lat_scat + lat_inc;
    if (theta > M_PI) {
      theta = 2.0 * M_PI - theta;
    }
  }

  Scalar sigma_1, sigma_2;

  //if (math::small(lat_inc) || math::equal(lat_inc, M_PI)) {
  //  sigma_1 = lon_scat - lon_inc;
  //  sigma_2 = 0.0;
  //} else if (math::small(lat_scat) || math::equal(lat_scat, M_PI)) {
  //  sigma_1 = 0.0;
  //  sigma_2 = lon_scat - lon_inc;
  //} else {
  //  sigma_1 = math::save_acos((cos(lat_scat) - cos(lat_inc) * cos_theta) /
  //                       (sin(lat_inc) * sin(theta)));
  //  sigma_2 = math::save_acos((cos(lat_inc) - cos(lat_scat) * cos_theta) /
  //                       (sin(lat_scat) * sin(theta)));
  //}
  if (math::small(lat_inc)) {
      sigma_1 = lon_scat - lon_inc;
      sigma_2 = 0.0;
  } else if (math::equal(lat_inc, M_PI)) {
      sigma_1 = lon_scat - lon_inc;
      sigma_2 = M_PI;
  } else if (math::small(lat_scat)) {
      sigma_1 = 0.0;
      sigma_2 = M_PI + lon_scat - lon_inc;
  } else if (math::equal(lat_scat, M_PI)) {
      sigma_1 = M_PI;
      sigma_2 = lon_scat - lon_inc;
  } else {
      sigma_1 = math::save_acos((cos(lat_scat) - cos(lat_inc) * cos_theta) /
                                (sin(lat_inc) * sin(theta)));
      sigma_2 = math::save_acos((cos(lat_inc) - cos(lat_scat) * cos_theta) /
                                (sin(lat_scat) * sin(theta)));
  }

  Scalar c_1 = cos(2.0 * sigma_1);
  Scalar c_2 = cos(2.0 * sigma_2);
  Scalar s_1 = sin(2.0 * sigma_1);
  Scalar s_2 = sin(2.0 * sigma_2);

  return {theta, c_1, c_2, s_1, s_2};
}

////////////////////////////////////////////////////////////////////////////////
// Manipulation of compact phase matrix format.
////////////////////////////////////////////////////////////////////////////////

template <typename Derived>
struct CompactFormatBase {
  template <typename TensorType, typename VectorType, typename MatrixType>
  static void expand_and_transform(TensorType &output,
                                   const TensorType &input,
                                   const VectorType &scat_angs,
                                   const VectorType &lon_scat,
                                   const MatrixType &rotation_coeffs) {
    using Scalar = typename TensorType::Scalar;
    using CoefficientVectorMap = Eigen::Map<
        const Eigen::Matrix<Scalar, 1, Derived::n_coeffs, Eigen::RowMajor>>;
    using PhaseMatrixMap = Eigen::Map< Eigen::Matrix<Scalar,
                                                          Derived::stokes_dim,
                                                          Derived::stokes_dim,
                                                          Eigen::RowMajor>>;

    eigen::IndexArray<6> dimensions_loop = {input.dimension(0),
                                            input.dimension(1),
                                            input.dimension(2),
                                            input.dimension(3),
                                            input.dimension(4),
                                            input.dimension(5)};
    eigen::Index scat_ang_index = 0;
    eigen::Index input_index = 0;
    eigen::Index output_index = 0;
    for (eigen::DimensionCounter<6> i{dimensions_loop}; i; ++i) {
      PhaseMatrixMap output_matrix(output.data() + output_index);
      CoefficientVectorMap input_vector(input.data() + input_index);

      scat_ang_index %= rotation_coeffs.rows();
      auto scat_ang = scat_angs[scat_ang_index];

      if (math::small(scat_ang) || math::equal(scat_ang, M_PI)) {
          Derived::expand(output_matrix, input_vector);
      } else {
        bool lon_scat_gt_pi = lon_scat[i.coordinates[4]] > M_PI;
        Derived::expand_and_transform(output_matrix,
                                      input_vector,
                                      rotation_coeffs(scat_ang_index, 0),
                                      rotation_coeffs(scat_ang_index, 1),
                                      rotation_coeffs(scat_ang_index, 2),
                                      rotation_coeffs(scat_ang_index, 3),
                                      lon_scat_gt_pi);
      }
      scat_ang_index++;
      output_index += output.dimension(6);
      input_index += input.dimension(6);
    }
  }
};

template <eigen::Index stokes_dim_> struct CompactFormat;

template <>
struct CompactFormat<4> : public CompactFormatBase<CompactFormat<4>> {

  using CompactFormatBase<CompactFormat<4>>::expand_and_transform;
  static constexpr eigen::Index n_coeffs = 6;
  static constexpr eigen::Index stokes_dim = 4;

  template <typename MatrixType, typename VectorType>
  static void expand(MatrixType &output, const VectorType &input) {
    output(0, 0) = input[0];
    output(0, 1) = f12(input);
    output(1, 0) = f12(input);
    output(1, 1) = f22(input);
    output(0, 2) = 0.0;
    output(1, 2) = 0.0;
    output(2, 0) = 0.0;
    output(2, 1) = 0.0;
    output(2, 2) = f33(input);
    output(0, 3) = 0.0;
    output(1, 3) = 0.0;
    output(2, 3) = f34(input);
    output(3, 0) = 0.0;
    output(3, 1) = 0.0;
    output(3, 2) = -f34(input);
    output(3, 3) = f44(input);
  }

  template <typename MatrixType, typename VectorType, typename Scalar>
  static void expand_and_transform(MatrixType &output,
                                   const VectorType &input,
                                   Scalar c_1,
                                   Scalar c_2,
                                   Scalar s_1,
                                   Scalar s_2,
                                   bool lon_scat_gt_pi) {
    output(0, 0) = input[0];

    output(0, 1) = c_1 * f12(input);
    output(1, 0) = c_2 * f12(input);
    output(1, 1) = c_1 * c_2 * f22(input) - s_1 * s_2 * f33(input);

    output(0, 2) = s_1 * f12(input);
    output(1, 2) = s_1 * c_2 * f22(input) + c_1 * s_2 * f33(input);
    output(2, 0) = -s_2 * f12(input);
    output(2, 1) = -c_1 * s_2 * f22(input) - s_1 * c_2 * f33(input);
    output(2, 2) = -s_1 * s_2 * f22(input) + c_1 * c_2 * f33(input);

    if (lon_scat_gt_pi) {
      output(0, 2) *= -1.0;
      output(1, 2) *= -1.0;
      output(2, 0) *= -1.0;
      output(2, 1) *= -1.0;
    }

    output(0, 3) = 0.0;
    output(1, 3) = s_2 * f34(input);
    output(2, 3) = c_2 * f34(input);
    output(3, 0) = 0.0;
    output(3, 1) = s_1 * f34(input);
    output(3, 2) = -c_1 * f34(input);
    output(3, 3) = f44(input);

    if (lon_scat_gt_pi) {
      output(1, 3) *= -1.0;
      output(3, 1) *= -1.0;
    }
  }

};

template <>
struct CompactFormat<3> : public CompactFormatBase<CompactFormat<3>> {

  using CompactFormatBase<CompactFormat<3>>::expand_and_transform;
  static constexpr eigen::Index n_coeffs = 6;
  static constexpr eigen::Index stokes_dim = 3;

  template <typename MatrixType, typename VectorType>
  static void expand(MatrixType &output, const VectorType &input) {
    output(0, 0) = input[0];
    output(0, 1) = f12(input);
    output(1, 0) = f12(input);
    output(1, 1) = f22(input);
    output(0, 2) = 0.0;
    output(1, 2) = 0.0;
    output(2, 0) = 0.0;
    output(2, 1) = 0.0;
    output(2, 2) = f33(input);
  }

  template <typename MatrixType, typename VectorType, typename Scalar>
  static void expand_and_transform(MatrixType &output,
                                   const VectorType &input,
                                   Scalar c_1,
                                   Scalar c_2,
                                   Scalar s_1,
                                   Scalar s_2,
                                   bool lon_scat_gt_pi) {

    output(0, 0) = input[0];

    output(0, 1) = c_1 * f12(input);
    output(1, 0) = c_2 * f12(input);
    output(1, 1) = c_1 * c_2 * f22(input) - s_1 * s_2 * f33(input);

    output(0, 2) = s_1 * f12(input);
    output(1, 2) = s_1 * c_2 * f22(input) + c_1 * s_2 * f33(input);
    output(2, 0) = -s_2 * f12(input);
    output(2, 1) = -c_1 * s_2 * f22(input) - s_1 * c_2 * f33(input);
    output(2, 2) = -s_1 * s_2 * f22(input) + c_1 * c_2 * f33(input);

    if (lon_scat_gt_pi) {
        output(0, 2) *= -1.0;
        output(1, 2) *= -1.0;
        output(2, 0) *= -1.0;
        output(2, 1) *= -1.0;
    }
  }
};

template <>
struct CompactFormat<2> : public CompactFormatBase<CompactFormat<2>> {

    using CompactFormatBase<CompactFormat<2>>::expand_and_transform;
    static constexpr eigen::Index n_coeffs = 4;
    static constexpr eigen::Index stokes_dim = 2;

  template <typename MatrixType, typename VectorType>
  static void expand(MatrixType &output, const VectorType &input) {
    output(0, 0) = input[0];
    output(0, 1) = f12(input);
    output(1, 0) = f12(input);
    output(1, 1) = f22(input);
  }

  template <typename MatrixType, typename VectorType, typename Scalar>
  static void expand_and_transform(MatrixType &output,
                                   const VectorType &input,
                                   Scalar c_1,
                                   Scalar c_2,
                                   Scalar s_1,
                                   Scalar s_2,
                                   bool /*lon_scat_gt_pi*/) {
    output(0, 0) = input[0];

    output(0, 1) = c_1 * f12(input);
    output(1, 0) = c_2 * f12(input);
    output(1, 1) = c_1 * c_2 * f22(input) - s_1 * s_2 * f33(input);
  }
};

template <>
struct CompactFormat<1> : public CompactFormatBase<CompactFormat<1>> {

  using CompactFormatBase<CompactFormat<1>>::expand_and_transform;
  static constexpr eigen::Index n_coeffs = 1;
  static constexpr eigen::Index stokes_dim = 1;

  template <typename MatrixType, typename VectorType>
  static void expand(MatrixType &output, const VectorType &input) {
    output(0, 0) = input[0];
  }

  template <typename MatrixType, typename VectorType, typename Scalar>
  static void expand_and_transform(MatrixType &output,
                                   const VectorType &input,
                                   Scalar /*c_1*/,
                                   Scalar /*c_2*/,
                                   Scalar /*s_1*/,
                                   Scalar /*s_2*/,
                                   bool /*lon_scat_gt_pi*/) {
    output(0, 0) = input[0];
  }
};

// pxx :: export
// pxx :: instance(PhaseMatrixDataGridded, ["scattering::ScatteringDataGridded<double>"])
template <typename Base>
    class PhaseMatrix : public Base {


  using typename Base::Coefficient;

  using typename Base::Vector;
  using typename Base::VectorPtr;
  using typename Base::DataTensor;
  using typename Base::DataTensorPtr;

  using Base::coeff_dim;
  using Base::data_;
  using Base::f_grid_;
  using Base::t_grid_;
  using Base::lon_inc_;
  using Base::lat_inc_;
  using Base::lon_scat_;
  using Base::lat_scat_;
  using Base::rank;

  using Base::n_temps_;
  using Base::n_freqs_;
  using Base::n_lon_inc_;
  using Base::n_lat_inc_;
  using Base::n_lon_scat_;
  using Base::n_lat_scat_;

 public:

  using PhaseFunction = eigen::Tensor<Coefficient, rank - 1>;
  using ScatteringMatrix = eigen::Tensor<Coefficient, rank + 1>;
  using Scalar = typename Base::Scalar;

  using Base::copy;

  /// Perfect forwarding constructor.
  template <typename... Args>
  PhaseMatrix(Args... args) : Base(std::forward<Args>(args)...) {}

  ParticleType get_particle_type() const {
      if (n_lon_inc_ > 1) {
          return ParticleType::General;
      }
      if ((n_lon_inc_ > 1) || (n_lat_inc_ > 1) || (n_lon_scat_ > 1)) {
          return ParticleType::AzimuthallyRandom;
      }
      return ParticleType::Random;
  }

  /// Determine stokes dimension of data.
  eigen::Index get_stokes_dim() const {
    auto n_coeffs = Base::get_n_coeffs();
    auto type = get_particle_type();
    if (type == ParticleType::Random) {
      if (n_coeffs == 6) {
        return 4;
      } else if (n_coeffs == 4) {
        return 3;
      }
      return 1;
    } else {
      return sqrt(n_coeffs);
    }
  }

  /// Reduce data to stokes dimension
  void set_stokes_dim(eigen::Index n) {
      auto stokes_dim_out = n;
      auto stokes_dim = get_stokes_dim();
      auto dimensions_new = data_->dimensions();

    if (get_particle_type() == ParticleType::Random) {
      if (stokes_dim_out == 1) {
        dimensions_new[coeff_dim] = 1;
      } else if (stokes_dim_out == 2) {
        dimensions_new[coeff_dim] = 4;
      } else {
        dimensions_new[coeff_dim] = 6;
      }
      auto data_new = std::make_shared<DataTensor>(dimensions_new);
      eigen::copy(*data_new, *data_);
      data_ = data_new;
    } else {
        dimensions_new[coeff_dim] = stokes_dim_out * stokes_dim_out;
        auto data_new = std::make_shared<DataTensor>(dimensions_new);

        auto stokes_dim_min = std::min(stokes_dim_out, stokes_dim);
        for (Index i = 0; i < stokes_dim_min; ++i) {
            for (Index j = 0; j < stokes_dim_min; ++j) {
                data_new->template chip<coeff_dim>(i * stokes_dim_out + j)
                    = data_->template chip<coeff_dim>(i * stokes_dim  + j);
            }
        }
        data_ = data_new;
    }
  }

  PhaseMatrix to_lab_frame(VectorPtr lat_inc_new,
                           VectorPtr lon_scat_new,
                           std::shared_ptr<LatitudeGrid<Scalar>> lat_scat_new,
                           Index stokes_dim) const {
    auto n_lat_inc = lat_inc_new->size();
    auto n_lon_scat = lon_scat_new->size();
    auto n_lat_scat = lat_scat_new->size();

    auto scat_ang_new = Vector(n_lat_inc * n_lon_scat * n_lat_scat);
    eigen::MatrixFixedRows<Scalar, 4> rotation_coeffs{
        n_lat_inc * n_lon_scat * n_lat_scat,
        4};

    Index index = 0;
    for (Index i = 0; i < n_lat_inc; ++i) {
      for (Index j = 0; j < n_lon_scat; ++j) {
        for (Index k = 0; k < n_lat_scat; ++k) {
          auto lat_inc = lat_inc_new->operator[](i);
          auto lon_scat = lon_scat_new->operator[](j);
          auto lat_scat = lat_scat_new->operator[](k);
          auto coeffs = rotation_coefficients(0.0, lat_inc, lon_scat, lat_scat);
          scat_ang_new[index] = coeffs[0];
          rotation_coeffs(index, 0) = coeffs[1];
          rotation_coeffs(index, 1) = coeffs[2];
          rotation_coeffs(index, 2) = coeffs[3];
          rotation_coeffs(index, 3) = coeffs[4];
          index += 1;
        }
      }
    }
    Index stokes_dim_in = get_stokes_dim();

    // Interpolate data to scattering angles.
    using Regridder = RegularRegridder<Scalar, 5>;
    Regridder regridder({*lat_scat_}, {scat_ang_new});
    eigen::IndexArray<7> dimensions_interp = {n_freqs_,
                                              n_temps_,
                                              1,
                                              n_lat_inc,
                                              n_lon_scat,
                                              n_lat_scat,
                                              Base::get_n_coeffs()};
    auto data_interp = regridder.regrid(*data_);
    data_interp.resize(dimensions_interp);

    stokes_dim = std::min(stokes_dim_in, stokes_dim);

    // Tensor to hold results.
    eigen::IndexArray<7> dimensions_new = {n_freqs_,
                                           n_temps_,
                                           1,
                                           n_lat_inc,
                                           n_lon_scat,
                                           n_lat_scat,
                                           stokes_dim * stokes_dim};
    auto data_new = std::make_shared<DataTensor>(dimensions_new);

    if (stokes_dim == 1) {
      CompactFormat<1>::expand_and_transform(*data_new,
                                             data_interp,
                                             scat_ang_new,
                                             *lon_scat_new,
                                             rotation_coeffs);
    } else if (stokes_dim == 2) {
      CompactFormat<2>::expand_and_transform(*data_new,
                                             data_interp,
                                             scat_ang_new,
                                             *lon_scat_new,
                                             rotation_coeffs);
    } else if (stokes_dim == 3) {
      CompactFormat<3>::expand_and_transform(*data_new,
                                             data_interp,
                                             scat_ang_new,
                                             *lon_scat_new,
                                             rotation_coeffs);
    } else {
      CompactFormat<4>::expand_and_transform(*data_new,
                                             data_interp,
                                             scat_ang_new,
                                             *lon_scat_new,
                                             rotation_coeffs);
    }
    return PhaseMatrix(f_grid_,
                       t_grid_,
                       lon_inc_,
                       lat_inc_new,
                       lon_scat_new,
                       lat_scat_new,
                       data_new);
  }

  PhaseMatrix to_lab_frame(Vector lat_inc,
                           Vector lon_scat,
                           Vector lat_scat,
                           Index stokes_dim) const {
    return to_lab_frame(std::make_shared<Vector>(lat_inc),
                        std::make_shared<Vector>(lon_scat),
                        std::make_shared<IrregularLatitudeGrid<Scalar>>(lat_scat),
                        stokes_dim);
  }

  PhaseMatrix to_lab_frame(Index n_lat_inc,
                           Index n_lon_scat,
                           Index stokes_dim) const {
      if ((n_lat_inc_ > 1) || (n_lon_scat_ > 1)) {
          return copy();
      }
      auto lon_scat_new =
          std::make_shared<Vector>(sht::SHT::get_longitude_grid(n_lon_scat));
      auto lat_scat_new =
          std::make_shared<sht::SHT::LatGrid>(sht::SHT::get_latitude_grid(n_lon_scat));
      auto lat_inc_new =
          std::make_shared<Vector>(sht::SHT::get_latitude_grid(n_lat_inc));
      return to_lab_frame(lat_inc_new, lon_scat_new, lat_scat_new, stokes_dim);
  }

  PhaseFunction get_phase_function() const { return data_->template chip<rank - 1>(0); }

  ScatteringMatrix get_scattering_matrix(Index stokes_dim) const {
    stokes_dim = std::min(stokes_dim, get_stokes_dim());
    auto output_dimensions = eigen::get_dimensions<rank + 1>(*data_);
    output_dimensions[rank - 1] = stokes_dim;
    output_dimensions[rank] = stokes_dim;
    return data_->reshape(output_dimensions);
  }
};

// pxx :: export
// pxx :: instance(PhaseMatrixDataGridded, ["scattering::ScatteringDataGridded<double>"])
template <typename Base>
class ExtinctionMatrix : public Base {


  using typename Base::Coefficient;

  using typename Base::Vector;
  using typename Base::DataTensor;
  using typename Base::DataTensorPtr;

  using Base::coeff_dim;
  using Base::data_;
  using Base::f_grid_;
  using Base::t_grid_;
  using Base::lon_inc_;
  using Base::lat_inc_;
  using Base::lon_scat_;
  using Base::lat_scat_;
  using Base::rank;

  using Base::n_temps_;
  using Base::n_freqs_;
  using Base::n_lon_inc_;
  using Base::n_lat_inc_;
  using Base::n_lon_scat_;
  using Base::n_lat_scat_;

 public:

  using ExtinctionCoefficient = eigen::Tensor<Coefficient, rank - 1>;
  using ExtinctionMatrixData = eigen::Tensor<Coefficient, rank + 1>;
  using Scalar = typename Base::Scalar;

  using Base::copy;

  /// Perfect forwarding constructor.
  template <typename... Args>
  ExtinctionMatrix(Args... args) : Base(std::forward<Args>(args)...) {}

  /// Determine stokes dimension of data.
  eigen::Index get_stokes_dim() const {
    auto n_coeffs = Base::get_n_coeffs();
    auto type = get_particle_type();
    if (type == ParticleType::Random) {
      return 4;
    } else {
      if (n_coeffs >= 3) {
        return 4;
      }
      if (n_coeffs >= 2) {
        return 2;
      }
      return 1;
    }
  }

  ParticleType get_particle_type() const {
    if (data_->dimension(coeff_dim) == 1) {
      return ParticleType::Random;
    } else {
      return ParticleType::AzimuthallyRandom;
    }
  }

  /// Reduce data to stokes dimension
  void set_stokes_dim(eigen::Index n) {
    auto stokes_dim = std::min(n, get_stokes_dim());
    auto dimensions_new = data_->dimensions();

    if (get_particle_type() == ParticleType::Random) {
      return;
    } else {
      if (stokes_dim == 1) {
        dimensions_new[coeff_dim] = 1;
      } else if (stokes_dim == 2) {
        dimensions_new[coeff_dim] = 2;
      } else if (stokes_dim == 3) {
        dimensions_new[coeff_dim] = 3;
      }
      auto data_new = std::make_shared<DataTensor>(dimensions_new);
      eigen::copy(*data_new, *data_);
      data_ = data_new;
    }
  }


  ExtinctionCoefficient get_extinction_coeff() const { return data_->template chip<rank - 1>(0); }

  ExtinctionMatrixData get_extinction_matrix(Index stokes_dim) const {
      std::array<eigen::Index, rank> dimensions = data_->dimensions();
      dimensions[rank - 1] = stokes_dim * stokes_dim;

      auto stokes_dim_min = std::min(stokes_dim, get_stokes_dim());


      auto new_data = eigen::Tensor<Coefficient, rank>(dimensions);
      new_data.setZero();

      auto type = get_particle_type();
      if (type == ParticleType::Random) {
          new_data.template chip<rank - 1>(0) = data_->template chip<rank-1>(0);
          if (stokes_dim >= 2) {
              new_data.template chip<rank - 1>(stokes_dim + 1) = data_->template chip<rank-1>(0);
          }
          if (stokes_dim >= 3) {
              new_data.template chip<rank - 1>(2 * stokes_dim + 2) = data_->template chip<rank-1>(0);
          }
          if (stokes_dim >= 4) {
              new_data.template chip<rank - 1>(3 * stokes_dim + 3) = data_->template chip<rank-1>(0);
          }
      } else {
          new_data.template chip<rank - 1>(0) = data_->template chip<rank-1>(0);
          if (stokes_dim >= 2) {
              new_data.template chip<rank - 1>(stokes_dim + 1) = data_->template chip<rank-1>(0);
              if (stokes_dim_min > 1) {
                new_data.template chip<rank - 1>(2) = data_->template chip<rank-1>(1);
                new_data.template chip<rank - 1>(stokes_dim) = data_->template chip<rank-1>(1);
              }
          }
          if (stokes_dim >= 3) {
              new_data.template chip<rank - 1>(2 * stokes_dim + 2) = data_->template chip<rank-1>(0);
          }
          if (stokes_dim >= 4) {
              new_data.template chip<rank - 1>(3 * stokes_dim + 3) = data_->template chip<rank-1>(0);
              if (stokes_dim_min > 2) {
                new_data.template chip<rank - 1>(2 * stokes_dim + 3) =
                    data_->template chip<rank - 1>(2);
                new_data.template chip<rank - 1>(3 * stokes_dim + 2) =
                    -data_->template chip<rank - 1>(2);
              }
          }
      }
      auto output_dimensions = eigen::get_dimensions<rank + 1>(*data_);
      output_dimensions[rank - 1] = stokes_dim;
      output_dimensions[rank] = stokes_dim;
      return new_data.reshape(output_dimensions);
  }
};

template <typename Base>
class AbsorptionVector : public Base {


  using typename Base::Coefficient;

  using typename Base::Vector;
  using typename Base::DataTensor;
  using typename Base::DataTensorPtr;

  using Base::coeff_dim;
  using Base::data_;
  using Base::f_grid_;
  using Base::t_grid_;
  using Base::lon_inc_;
  using Base::lat_inc_;
  using Base::lon_scat_;
  using Base::lat_scat_;
  using Base::rank;

  using Base::n_temps_;
  using Base::n_freqs_;
  using Base::n_lon_inc_;
  using Base::n_lat_inc_;
  using Base::n_lon_scat_;
  using Base::n_lat_scat_;

 public:

  using AbsorptionCoefficient = eigen::Tensor<Coefficient, rank - 1>;
  using AbsorptionVectorData = eigen::Tensor<Coefficient, rank>;
  using Scalar = typename Base::Scalar;

  using Base::copy;

  /// Perfect forwarding constructor.
  template <typename... Args>
  AbsorptionVector(Args... args) : Base(std::forward<Args>(args)...) {}

  ParticleType get_particle_type() const {
    if (data_->dimension(coeff_dim) == 1) {
      return ParticleType::Random;
    } else {
      return ParticleType::AzimuthallyRandom;
    }
  }

  /// Determine stokes dimension of data.
  eigen::Index get_stokes_dim() const {
    auto n_coeffs = Base::get_n_coeffs();
    auto type = get_particle_type();
    if (type == ParticleType::Random) {
      return 4;
    } else {
      if (n_coeffs >= 3) {
        return 4;
      }
      if (n_coeffs >= 2) {
        return 2;
      }
      return 1;
    }
  }

  /// Reduce data to stokes dimension
  void set_stokes_dim(eigen::Index n) {
    auto stokes_dim = std::min(n, get_stokes_dim());
    auto dimensions_new = data_->dimensions();

    if (get_particle_type() == ParticleType::Random) {
      return;
    } else {
      if (stokes_dim == 1) {
        dimensions_new[coeff_dim] = 1;
      } else if (stokes_dim >= 2) {
        dimensions_new[coeff_dim] = 2;
      }
      auto data_new = std::make_shared<DataTensor>(dimensions_new);
      eigen::copy(*data_new, *data_);
      data_ = data_new;
    }
  }


  AbsorptionCoefficient get_absorption_coeff() const { return data_->template chip<rank - 1>(0); }

  AbsorptionVectorData get_absorption_vector(Index stokes_dim) const {
      std::array<eigen::Index, rank> dimensions = data_->dimensions();
      dimensions[rank - 1] = stokes_dim;
      stokes_dim = std::min(get_stokes_dim(), stokes_dim);

      auto new_data = eigen::Tensor<Coefficient, rank>(dimensions);
      new_data.setZero();

      new_data.template chip<rank - 1>(0) = data_->template chip<rank-1>(0);
      auto type = get_particle_type();
      if ((type == ParticleType::AzimuthallyRandom) && (stokes_dim > 1)) {
          new_data.template chip<rank - 1>(1) = data_->template chip<rank-1>(1);
      }
      return new_data;
  }
};



}  // namespace stokes
}  // namespace scattering
#endif
