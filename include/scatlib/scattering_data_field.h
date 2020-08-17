/** \file scattering_data.h
 *
 * Represents scalar data that is defined on the product
 * space of two solid angles.
 *
 * @author Simon Pfreundschuh, 2020
 */
#ifndef __SCATLIB_SCATTERING_DATA_FIELD__
#define __SCATLIB_SCATTERING_DATA_FIELD__

#include <scatlib/eigen.h>
#include <scatlib/sht.h>
#include <scatlib/interpolation.h>
#include <memory>
#include <cassert>

namespace scatlib {
namespace scattering_data {

enum class DataFormat { Gridded, Spectral, FullySpectral };
enum class DataType { Spherical, TotallyRandom, AzimuthallyRandom, General };


class ScatteringDataFieldBase {
  static DataType determine_type(size_t n_lon_inc,
                                 size_t n_lat_inc,
                                 size_t n_lon_scat,
                                 size_t n_lat_scat) {
    if ((n_lon_inc == 1) && (n_lat_inc == 1) &&
        (n_lon_scat == 1) && (n_lat_scat == 1)) {
      return DataType::Spherical;
    }
    if ((n_lon_inc == 1) && (n_lat_inc == 1) &&
        (n_lon_scat == 1)) {
      return DataType::TotallyRandom;
    }
    if (n_lon_inc == 1) {
      return DataType::AzimuthallyRandom;
    }
    return DataType::General;
  }

public:
ScatteringDataFieldBase(size_t n_freqs,
                        size_t n_temps,
                        size_t n_lon_inc,
                          size_t n_lat_inc,
                          size_t n_lon_scat,
                          size_t n_lat_scat)
    : n_freqs_(n_freqs),
      n_temps_(n_temps),
      n_lon_inc_(n_lon_inc),
        n_lat_inc_(n_lat_inc),
        n_lon_scat_(n_lon_scat),
        n_lat_scat_(n_lat_scat),
        type_(determine_type(n_lon_inc, n_lat_inc, n_lon_scat, n_lat_scat)) {}

  DataType get_type() const { return type_; }

 protected:
  size_t n_freqs_;
  size_t n_temps_;
  size_t n_lon_inc_;
  size_t n_lat_inc_;
  size_t n_lon_scat_;
  size_t n_lat_scat_;
  DataType type_;
};

template <typename Scalar, DataFormat type>
    class ScatteringDataField;

// pxx :: export
// pxx :: instance(["double"])
template <typename Scalar>
class ScatteringDataFieldGridded
    : public ScatteringDataFieldBase {
 public:
  using ScatteringDataFieldBase::get_type;
  using ScatteringDataFieldBase::n_freqs_;
  using ScatteringDataFieldBase::n_temps_;
  using ScatteringDataFieldBase::n_lat_inc_;
  using ScatteringDataFieldBase::n_lat_scat_;
  using ScatteringDataFieldBase::n_lon_inc_;
  using ScatteringDataFieldBase::n_lon_scat_;
  using ScatteringDataFieldBase::type_;

  using Vector = eigen::Vector<Scalar>;
  using VectorMap = eigen::VectorMap<Scalar>;
  using SharedVectorPtr = const std::shared_ptr<const eigen::Vector<Scalar>>;
  using ConstVectorMap = eigen::ConstVectorMap<Scalar>;
  using Matrix = eigen::Matrix<Scalar>;
  using MatrixMap = eigen::MatrixMap<Scalar>;
  using ConstMatrixMap = eigen::ConstMatrixMap<Scalar>;
  using OneAngle = eigen::MatrixFixedRows<Scalar, 1>;
  using ThreeAngles = eigen::MatrixFixedRows<Scalar, 3>;
  using FourAngles = eigen::MatrixFixedRows<Scalar, 4>;

  template <eigen::Index rank>
  using Tensor = eigen::Tensor<Scalar, rank>;
  template <eigen::Index rank>
  using TensorMap = eigen::TensorMap<Scalar, rank>;
  template <eigen::Index rank>
  using ConstTensorMap = eigen::ConstTensorMap<Scalar, rank>;
  using DataTensor = eigen::Tensor<Scalar, 7>;
  using SharedDataTensorPtr = const std::shared_ptr<const DataTensor>;

  // pxx :: hide
  ScatteringDataFieldGridded(SharedVectorPtr f_grid,
                             SharedVectorPtr t_grid,
                             SharedVectorPtr lon_inc,
                             SharedVectorPtr lat_inc,
                             SharedVectorPtr lon_scat,
                             SharedVectorPtr lat_scat,
                             SharedDataTensorPtr data)
      : ScatteringDataFieldBase(f_grid->size(),
                                t_grid->size(),
                                lon_inc->size(),
                                lat_inc->size(),
                                lon_scat->size(),
                                lat_scat->size()),
        f_grid_(f_grid),
        t_grid_(t_grid),
        lon_inc_(lon_inc),
        lat_inc_(lat_inc),
        lon_scat_(lon_scat),
        lat_scat_(lat_scat),
        f_grid_map_(f_grid->data(), n_freqs_),
        t_grid_map_(t_grid->data(), n_temps_),
        lon_inc_map_(lon_inc->data(), n_lon_inc_),
        lat_inc_map_(lat_inc->data(), n_lat_inc_),
        lon_scat_map_(lon_scat->data(), n_lon_scat_),
        lat_scat_map_(lat_scat->data(), n_lat_scat_),
        data_(data) {}

  ScatteringDataFieldGridded(Vector f_grid,
                             Vector t_grid,
                             Vector &lon_inc,
                             Vector &lat_inc,
                             Vector &lon_scat,
                             Vector &lat_scat,
                             Tensor<7> &data)
      : ScatteringDataFieldBase(f_grid.size(),
                                t_grid.size(),
                                lon_inc.size(),
                                lat_inc.size(),
                                lon_scat.size(),
                                lat_scat.size()),
        f_grid_(std::make_shared<Vector>(f_grid)),
        t_grid_(std::make_shared<Vector>(t_grid)),
        lon_inc_(std::make_shared<Vector>(lon_inc)),
        lat_inc_(std::make_shared<Vector>(lat_inc)),
        lon_scat_(std::make_shared<Vector>(lon_scat)),
        lat_scat_(std::make_shared<Vector>(lat_scat)),
        data_(std::make_shared<DataTensor>(data)),
        f_grid_map_(f_grid_->data(), n_freqs_),
        t_grid_map_(t_grid_->data(), n_temps_),
        lon_inc_map_(lon_inc_->data(), n_lon_inc_),
        lat_inc_map_(lat_inc_->data(), n_lat_inc_),
        lon_scat_map_(lon_scat_->data(), n_lon_scat_),
        lat_scat_map_(lat_scat_->data(), n_lat_scat_) {}

  // pxx :: hide
  ScatteringDataFieldGridded interpolate_frequency(
      std::shared_ptr<Vector> frequencies) const {
    using Interpolator = RegularGridInterpolator<DataTensor, 1, ConstVectorMap>;
    Interpolator interpolator({f_grid_map_});
    auto dimensions_new = data_->dimensions();
    dimensions_new[0] = frequencies->size();
    auto data_new = std::make_shared<DataTensor>(dimensions_new);
    data_new->setConstant(0.0);
    auto weights = interpolator.calculate_weights(*frequencies);
    auto result_map = TensorMap<7>(data_new->data(), data_new->dimensions());
    interpolator.template interpolate(result_map, *data_, weights);
    auto result = ScatteringDataFieldGridded(frequencies,
                                             t_grid_,
                                             lon_inc_,
                                             lat_inc_,
                                             lon_scat_,
                                             lat_scat_,
                                             data_new);
    return result;
  }

  ScatteringDataFieldGridded interpolate_frequency(const Vector &frequencies) const {
    return interpolate_frequency(std::make_shared<Vector>(frequencies));
  }

  // pxx :: hide
  ScatteringDataFieldGridded interpolate_temperature(
      std::shared_ptr<Vector> temperatures) const {
      using Interpolator = RegularGridInterpolator<typename eigen::IndexResult<const DataTensor, 1>::type, 1, ConstVectorMap>;
      Interpolator interpolator({t_grid_map_});
      auto dimensions_new = data_->dimensions();
      dimensions_new[1] = temperatures->size();
      auto data_new = std::make_shared<DataTensor>(DataTensor(dimensions_new));
      auto weights = interpolator.calculate_weights(*temperatures);
      auto interpolate = [&interpolator, &weights](const TensorMap<6> &out,
                                                   const ConstTensorMap<6> &in) {
          interpolator.interpolate(out, in, weights);
      };
      eigen::map_over_dimensions<1>(*data_new, *data_, interpolate);
      return ScatteringDataFieldGridded(f_grid_,
                                        temperatures,
                                        lon_inc_,
                                        lat_inc_,
                                        lon_scat_,
                                        lat_scat_,
                                        data_new);
  }

  ScatteringDataFieldGridded interpolate_temperature(const Vector &temperatures) const {
      return interpolate_temperature(std::make_shared<Vector>(temperatures));
  }

  // pxx :: hide
  ScatteringDataFieldGridded interpolate_angles(
      SharedVectorPtr lon_inc_new,
      SharedVectorPtr lat_inc_new,
      SharedVectorPtr lon_scat_new,
      SharedVectorPtr lat_scat_new)
       const {
      using Regridder = RegularRegridder<Scalar, 5, 4>;
      Regridder regridder({*lon_inc_, *lat_inc_, *lon_scat_, *lat_scat_},
                          {*lon_inc_new, *lat_inc_new, *lon_scat_new, *lat_scat_new},
                          {0, 1, 2, 3});
      auto dimensions_new = data_->dimensions();
      dimensions_new[2] = lon_inc_new->size();
      dimensions_new[3] = lat_inc_new->size();
      dimensions_new[4] = lon_scat_new->size();
      dimensions_new[5] = lat_scat_new->size();
      auto data_new = std::make_shared<DataTensor>(DataTensor(dimensions_new));
      auto regrid = [&regridder](const TensorMap<5> &out,
                                 const ConstTensorMap<5> &in) {
          return regridder.regrid(out, in);
      };
      eigen::map_over_dimensions<2>(*data_new, *data_, regrid);
      return ScatteringDataFieldGridded(f_grid_,
                                        t_grid_,
                                        lon_inc_new,
                                        lat_inc_new,
                                        lon_scat_new,
                                        lat_scat_new,
                                        data_new);
  }

  ScatteringDataFieldGridded interpolate_angles(
      Vector lon_inc_new,
      Vector lat_inc_new,
      Vector lon_scat_new,
      Vector lat_scat_new) const {
      return interpolate_angles(std::make_shared<const Vector>(lon_inc_new),
                                std::make_shared<const Vector>(lat_inc_new),
                                std::make_shared<const Vector>(lon_scat_new),
                                std::make_shared<const Vector>(lat_scat_new));
  }

  const DataTensor &get_data() const {return *data_;}

 protected:

  SharedVectorPtr f_grid_;
  SharedVectorPtr t_grid_;
  SharedVectorPtr lon_inc_;
  SharedVectorPtr lat_inc_;
  SharedVectorPtr lon_scat_;
  SharedVectorPtr lat_scat_;

  ConstVectorMap f_grid_map_;
  ConstVectorMap t_grid_map_;
  ConstVectorMap lon_inc_map_;
  ConstVectorMap lat_inc_map_;
  ConstVectorMap lon_scat_map_;
  ConstVectorMap lat_scat_map_;

  SharedDataTensorPtr data_;
};


}  // namespace scattering_data
}  // namespace scatlib

#endif
