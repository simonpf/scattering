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

// pxx :: hide
template <typename Scalar>
    class ScatteringDataFieldGridded;
template <typename Scalar>
    class ScatteringDataFieldSpectral;
template <typename Scalar>
    class ScatteringDataFieldFullySpectral;

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
    using Regridder = RegularRegridder<Scalar, 0>;
    Regridder regridder({*f_grid_}, {*frequencies});
    auto dimensions_new = data_->dimensions();
    auto data_interp = regridder.regrid(*data_);
    dimensions_new[0] = frequencies->size();
    auto data_new = std::make_shared<DataTensor>(std::move(data_interp));
    return ScatteringDataFieldGridded(frequencies,
                                      t_grid_,
                                      lon_inc_,
                                      lat_inc_,
                                      lon_scat_,
                                      lat_scat_,
                                      data_new);
  }

  ScatteringDataFieldGridded interpolate_frequency(const Vector &frequencies) const {
    return interpolate_frequency(std::make_shared<Vector>(frequencies));
  }

  // pxx :: hide
  ScatteringDataFieldGridded interpolate_temperature(
      std::shared_ptr<Vector> temperatures) const {
      using Regridder = RegularRegridder<Scalar, 1>;
      Regridder regridder({*t_grid_},
                          {*temperatures});
      auto dimensions_new = data_->dimensions();
      auto data_interp = regridder.regrid(*data_);
      dimensions_new[1] = temperatures->size();
      auto data_new = std::make_shared<DataTensor>(std::move(data_interp));
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
      using Regridder = RegularRegridder<Scalar, 2, 3, 4, 5>;
      Regridder regridder({*lon_inc_, *lat_inc_, *lon_scat_, *lat_scat_},
                          {*lon_inc_new, *lat_inc_new, *lon_scat_new, *lat_scat_new});
      auto dimensions_new = data_->dimensions();
      auto data_interp = regridder.regrid(*data_);
      auto data_new = std::make_shared<DataTensor>(std::move(data_interp));
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

  // pxx :: hide
  ScatteringDataFieldSpectral<Scalar> to_spectral(std::shared_ptr<sht::SHT> sht);

  ScatteringDataFieldSpectral<Scalar> to_spectral(size_t l_max,
                                                  size_t m_max) {

      std::shared_ptr<sht::SHT> sht = std::make_shared<sht::SHT>(l_max,
                                                                 m_max,
                                                                 n_lat_scat_,
                                                                 n_lon_scat_);
      return to_spectral(sht);
  }

  ScatteringDataFieldSpectral<Scalar> to_spectral() {
    size_t l_max = ((n_lat_scat_ % 2) == 0) ? n_lat_scat_ - 2 : n_lat_scat_ - 1;
    size_t m_max = (m_max > 2) ? (n_lon_inc_ / 2) - 1 : 0;
    return to_spectral(l_max, m_max);
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

// pxx :: export
// pxx :: instance(["double"])
template <typename Scalar>
class ScatteringDataFieldSpectral
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
  using SharedShtPtr = std::shared_ptr<sht::SHT>;

  template <eigen::Index rank>
  using CmplxTensor = eigen::Tensor<std::complex<Scalar>, rank>;
  template <eigen::Index rank>
  using CmplxTensorMap = eigen::TensorMap<std::complex<Scalar>, rank>;
  template <eigen::Index rank>
  using ConstCmplxTensorMap = eigen::ConstTensorMap<std::complex<Scalar>, rank>;
  using DataTensor = eigen::Tensor<std::complex<Scalar>, 6>;
  using SharedDataTensorPtr = const std::shared_ptr<const DataTensor>;

  // pxx :: hide
  ScatteringDataFieldSpectral(SharedVectorPtr f_grid,
                              SharedVectorPtr t_grid,
                              SharedVectorPtr lon_inc,
                              SharedVectorPtr lat_inc,
                              SharedShtPtr sht_scat,
                              SharedDataTensorPtr data)
      : ScatteringDataFieldBase(f_grid->size(),
                                t_grid->size(),
                                lon_inc->size(),
                                lat_inc->size(),
                                sht_scat->get_n_longitudes(),
                                sht_scat->get_n_latitudes()),
        f_grid_(f_grid),
        t_grid_(t_grid),
        lon_inc_(lon_inc),
        lat_inc_(lat_inc),
        sht_scat_(sht_scat),
        f_grid_map_(f_grid->data(), n_freqs_),
        t_grid_map_(t_grid->data(), n_temps_),
        lon_inc_map_(lon_inc->data(), n_lon_inc_),
        lat_inc_map_(lat_inc->data(), n_lat_inc_),
        data_(data) {}

  // pxx :: hide
  ScatteringDataFieldSpectral interpolate_frequency(
      std::shared_ptr<Vector> frequencies) const {
      using Regridder = RegularRegridder<Scalar, 0>;
      Regridder regridder({*f_grid_}, {*frequencies});
      auto dimensions_new = data_->dimensions();
      auto data_interp = regridder.regrid(*data_);
      dimensions_new[0] = frequencies->size();
      auto data_new = std::make_shared<DataTensor>(std::move(data_interp));
      return ScatteringDataFieldSpectral(frequencies,
                                         t_grid_,
                                        lon_inc_,
                                        lat_inc_,
                                        sht_scat_,
                                        data_new);
  }

  ScatteringDataFieldSpectral interpolate_frequency(const Vector &frequencies) const {
    return interpolate_frequency(std::make_shared<Vector>(frequencies));
  }

  // pxx :: hide
  ScatteringDataFieldSpectral interpolate_temperature(
      std::shared_ptr<Vector> temperatures) const {
      using Regridder = RegularRegridder<Scalar, 1>;
      Regridder regridder({*t_grid_},
                          {*temperatures});
      auto dimensions_new = data_->dimensions();
      auto data_interp = regridder.regrid(*data_);
      dimensions_new[1] = temperatures->size();
      auto data_new = std::make_shared<DataTensor>(std::move(data_interp));;
      return ScatteringDataFieldSpectral(f_grid_,
                                        temperatures,
                                        lon_inc_,
                                        lat_inc_,
                                        sht_scat_,
                                        data_new);
  }

  ScatteringDataFieldSpectral interpolate_temperature(const Vector &temperatures) const {
      return interpolate_temperature(std::make_shared<Vector>(temperatures));
  }

  // pxx :: hide
  ScatteringDataFieldSpectral interpolate_angles(
      SharedVectorPtr lon_inc_new,
      SharedVectorPtr lat_inc_new) const {
      using Regridder = RegularRegridder<Scalar, 2, 3>;
    Regridder regridder(
        {*lon_inc_, *lat_inc_},
        {*lon_inc_new, *lat_inc_new}
        );
    auto dimensions_new = data_->dimensions();
    dimensions_new[2] = lon_inc_new->size();
    dimensions_new[3] = lat_inc_new->size();
    auto data_new = std::make_shared<DataTensor>(DataTensor(dimensions_new));
    regridder.regrid(*data_new, *data_);
    return ScatteringDataFieldSpectral(f_grid_,
                                       t_grid_,
                                       lon_inc_new,
                                       lat_inc_new,
                                       sht_scat_,
                                       data_new);
  }

  ScatteringDataFieldSpectral interpolate_angles(
      Vector lon_inc_new,
      Vector lat_inc_new) const {
      return interpolate_angles(std::make_shared<const Vector>(lon_inc_new),
                                std::make_shared<const Vector>(lat_inc_new));
  }

  ScatteringDataFieldGridded<Scalar> to_gridded();
  ScatteringDataFieldFullySpectral<Scalar> to_fully_spectral(SharedShtPtr sht);
  ScatteringDataFieldFullySpectral<Scalar> to_fully_spectral(size_t l_max,
                                                             size_t m_max) {
    std::shared_ptr<sht::SHT> sht =
        std::make_shared<sht::SHT>(l_max, m_max, n_lat_inc_, n_lon_inc_);
    return to_fully_spectral(sht);
  }
  ScatteringDataFieldFullySpectral<Scalar> to_fully_spectral() {
      size_t l_max = ((n_lat_inc_ % 2) == 0) ? n_lat_inc_ - 2 : n_lat_inc_ - 1;
      size_t m_max = (m_max > 2) ? (n_lon_inc_ / 2) - 1 : 0;
      return to_fully_spectral(l_max, m_max);
  }

  const DataTensor &get_data() const {return *data_;}

 protected:

  SharedVectorPtr f_grid_;
  SharedVectorPtr t_grid_;
  SharedVectorPtr lon_inc_;
  SharedVectorPtr lat_inc_;
  SharedShtPtr sht_scat_;

  ConstVectorMap f_grid_map_;
  ConstVectorMap t_grid_map_;
  ConstVectorMap lon_inc_map_;
  ConstVectorMap lat_inc_map_;

  SharedDataTensorPtr data_;
};

// pxx :: export
// pxx :: instance(["double"])
template <typename Scalar>
class ScatteringDataFieldFullySpectral
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
  using SharedShtPtr = std::shared_ptr<sht::SHT>;

  template <eigen::Index rank>
  using CmplxTensor = eigen::Tensor<std::complex<Scalar>, rank>;
  template <eigen::Index rank>
  using CmplxTensorMap = eigen::TensorMap<std::complex<Scalar>, rank>;
  template <eigen::Index rank>
  using ConstCmplxTensorMap = eigen::ConstTensorMap<std::complex<Scalar>, rank>;
  using DataTensor = eigen::Tensor<std::complex<Scalar>, 5>;
  using SharedDataTensorPtr = const std::shared_ptr<const DataTensor>;

  // pxx :: hide
  ScatteringDataFieldFullySpectral(SharedVectorPtr f_grid,
                                   SharedVectorPtr t_grid,
                                   SharedShtPtr sht_inc,
                                   SharedShtPtr sht_scat,
                                   SharedDataTensorPtr data) : ScatteringDataFieldBase(f_grid->size(),
                                t_grid->size(),
                                sht_inc->get_n_longitudes(),
                                sht_inc->get_n_latitudes(),
                                sht_scat->get_n_longitudes(),
                                sht_scat->get_n_latitudes()),
        f_grid_(f_grid),
        t_grid_(t_grid),
        sht_inc_(sht_inc),
        sht_scat_(sht_scat),
        f_grid_map_(f_grid->data(), n_freqs_),
        t_grid_map_(t_grid->data(), n_temps_),
        data_(data) {}

  // pxx :: hide
  ScatteringDataFieldFullySpectral interpolate_frequency(
      std::shared_ptr<Vector> frequencies) const {
      using Regridder = RegularRegridder<Scalar, 0>;
      Regridder regridder({*f_grid_}, {*frequencies});
      auto dimensions_new = data_->dimensions();
      auto data_interp = regridder.regrid(*data_);
      dimensions_new[0] = frequencies->size();
      auto data_new = std::make_shared<DataTensor>(std::move(data_interp));
      return ScatteringDataFieldFullySpectral(frequencies,
                                              t_grid_,
                                              sht_inc_,
                                              sht_scat_,
                                              data_new);
  }

  ScatteringDataFieldFullySpectral interpolate_frequency(
      const Vector &frequencies) const {
    return interpolate_frequency(std::make_shared<Vector>(frequencies));
  }

  // pxx :: hide
  ScatteringDataFieldFullySpectral interpolate_temperature(
      std::shared_ptr<Vector> temperatures) const {
      using Regridder = RegularRegridder<Scalar, 1>;
      Regridder regridder({*t_grid_},
                          {*temperatures});
      auto dimensions_new = data_->dimensions();
      auto data_interp = regridder.regrid(*data_);
      dimensions_new[1] = temperatures->size();
      auto data_new = std::make_shared<DataTensor>(std::move(data_interp));;
      return ScatteringDataFieldFullySpectral(f_grid_,
                                              temperatures,
                                              sht_inc_,
                                              sht_scat_,
                                              data_new);
  }

  ScatteringDataFieldFullySpectral interpolate_temperature(const Vector &temperatures) const {
      return interpolate_temperature(std::make_shared<Vector>(temperatures));
  }

  ScatteringDataFieldSpectral<Scalar> to_spectral();

  const DataTensor &get_data() const {return *data_;}

 protected:

  SharedVectorPtr f_grid_;
  SharedVectorPtr t_grid_;
  SharedVectorPtr lon_inc_;
  SharedVectorPtr lat_inc_;
  SharedShtPtr sht_inc_;
  SharedShtPtr sht_scat_;

  ConstVectorMap f_grid_map_;
  ConstVectorMap t_grid_map_;

  SharedDataTensorPtr data_;
};


template <typename Scalar>
ScatteringDataFieldSpectral<Scalar>
ScatteringDataFieldGridded<Scalar>::to_spectral(std::shared_ptr<sht::SHT> sht) {
  eigen::IndexArray<5> dimensions_loop = {n_freqs_,
                                         n_temps_,
                                         n_lon_inc_,
                                         n_lat_inc_,
                                         data_->dimension(6)};
  eigen::IndexArray<6> dimensions_new = {n_freqs_,
                                         n_temps_,
                                         n_lon_inc_,
                                         n_lat_inc_,
                                         sht->get_n_spectral_coeffs(),
                                         data_->dimension(6)};
  using CmplxDataTensor = eigen::Tensor<std::complex<Scalar>, 6>;
  auto data_new = std::make_shared<CmplxDataTensor>(dimensions_new);
  for (eigen::DimensionCounter<5> i{dimensions_loop}; i; ++i) {
    eigen::get_subvector<4>(*data_new, i.coordinates) =
        sht->transform(eigen::get_submatrix<4, 5>(*data_, i.coordinates));
  }
  return ScatteringDataFieldSpectral<Scalar>(f_grid_,
                                             t_grid_,
                                             lon_inc_,
                                             lat_inc_,
                                             sht,
                                             data_new);
}

template <typename Scalar>
ScatteringDataFieldGridded<Scalar>
ScatteringDataFieldSpectral<Scalar>::to_gridded() {
  eigen::IndexArray<5> dimensions_loop = {n_freqs_,
                                          n_temps_,
                                          n_lon_inc_,
                                          n_lat_inc_,
                                          data_->dimension(5)};
  eigen::IndexArray<7> dimensions_new = {n_freqs_,
                                         n_temps_,
                                         n_lon_inc_,
                                         n_lat_inc_,
                                         sht_scat_->get_n_longitudes(),
                                         sht_scat_->get_n_latitudes(),
                                         data_->dimension(5)};
  using Vector = eigen::Vector<Scalar>;
  using DataTensor = eigen::Tensor<Scalar, 7>;
  auto data_new = std::make_shared<DataTensor>(dimensions_new);
  for (eigen::DimensionCounter<5> i{dimensions_loop}; i; ++i) {
      eigen::get_submatrix<4, 5>(*data_new, i.coordinates) =
        sht_scat_->synthesize(eigen::get_subvector<4>(*data_, i.coordinates));
  }
  auto lon_scat_ = std::make_shared<Vector>(sht_scat_->get_longitude_grid());
  auto lat_scat_ = std::make_shared<Vector>(sht_scat_->get_latitude_grid());
  return ScatteringDataFieldGridded<Scalar>(f_grid_,
                                            t_grid_,
                                            lon_inc_,
                                            lat_inc_,
                                            lon_scat_,
                                            lat_scat_,
                                            data_new);
}

template <typename Scalar>
ScatteringDataFieldFullySpectral<Scalar>
ScatteringDataFieldSpectral<Scalar>::to_fully_spectral(std::shared_ptr<sht::SHT> sht) {
  eigen::IndexArray<4> dimensions_loop = {n_freqs_,
                                          n_temps_,
                                          data_->dimension(4),
                                          data_->dimension(5)};
  eigen::IndexArray<5> dimensions_new = {n_freqs_,
                                         n_temps_,
                                         sht->get_n_spectral_coeffs_cmplx(),
                                         data_->dimension(4),
                                         data_->dimension(5)};
  using CmplxDataTensor = eigen::Tensor<std::complex<Scalar>, 5>;
  auto data_new = std::make_shared<CmplxDataTensor>(dimensions_new);
  for (eigen::DimensionCounter<4> i{dimensions_loop}; i; ++i) {
    eigen::get_subvector<2>(*data_new, i.coordinates) =
        sht->transform_cmplx(eigen::get_submatrix<2, 3>(*data_, i.coordinates));
  }
  return ScatteringDataFieldFullySpectral<Scalar>(f_grid_,
                                                  t_grid_,
                                                  sht,
                                                  sht_scat_,
                                                  data_new);
}

template <typename Scalar>
ScatteringDataFieldSpectral<Scalar>
ScatteringDataFieldFullySpectral<Scalar>::to_spectral() {
  eigen::IndexArray<4> dimensions_loop = {n_freqs_,
                                          n_temps_,
                                          data_->dimension(3),
                                          data_->dimension(4)};
  eigen::IndexArray<6> dimensions_new = {n_freqs_,
                                         n_temps_,
                                         sht_inc_->get_n_longitudes(),
                                         sht_inc_->get_n_latitudes(),
                                         data_->dimension(3),
                                         data_->dimension(4)};
  using CmplxDataTensor = eigen::Tensor<std::complex<Scalar>, 6>;
  auto data_new = std::make_shared<CmplxDataTensor>(dimensions_new);
  for (eigen::DimensionCounter<4> i{dimensions_loop}; i; ++i) {
    eigen::get_submatrix<2, 3>(*data_new, i.coordinates) =
        sht_inc_->transform_cmplx(eigen::get_subvector<2>(*data_, i.coordinates));
  }

  auto lon_inc_ = std::make_shared<Vector>(sht_inc_->get_longitude_grid());
  auto lat_inc_ = std::make_shared<Vector>(sht_inc_->get_latitude_grid());

  return ScatteringDataFieldSpectral<Scalar>(f_grid_,
                                             t_grid_,
                                             lon_inc_,
                                             lat_inc_,
                                             sht_scat_,
                                             data_new);
}

}  // namespace scattering_data
}  // namespace scatlib

#endif
