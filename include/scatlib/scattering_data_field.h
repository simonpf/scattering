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

using eigen::Index;

enum class DataFormat { Gridded, Spectral, FullySpectral };
enum class DataType { Spherical, TotallyRandom, AzimuthallyRandom, General };

// pxx :: hide
template <typename Scalar>
    class ScatteringDataFieldGridded;
template <typename Scalar>
    class ScatteringDataFieldSpectral;
template <typename Scalar>
    class ScatteringDataFieldFullySpectral;

/** ScatteringDataField base class.
 *
 * Holds information on the size of the angular grids and the
 * the type of scattering data.
 */
class ScatteringDataFieldBase {
    /** Determine scattering data type
     *
     * Determines the type of scattering data for a given phase matrix
     * tensor.
     */
  static DataType determine_type(Index n_lon_inc,
                                 Index n_lat_inc,
                                 Index n_lon_scat,
                                 Index n_lat_scat) {
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

protected:
 ScatteringDataFieldBase(Index n_freqs,
                         Index n_temps,
                         Index n_lon_inc,
                         Index n_lat_inc,
                         Index n_lon_scat,
                         Index n_lat_scat)
     : n_freqs_(n_freqs),
       n_temps_(n_temps),
       n_lon_inc_(n_lon_inc),
       n_lat_inc_(n_lat_inc),
       n_lon_scat_(n_lon_scat),
       n_lat_scat_(n_lat_scat),
       type_(determine_type(n_lon_inc, n_lat_inc, n_lon_scat, n_lat_scat)) {}

 DataType get_type() const { return type_; }

protected:
 Index n_freqs_;
 Index n_temps_;
 Index n_lon_inc_;
 Index n_lat_inc_;
 Index n_lon_scat_;
 Index n_lat_scat_;
 DataType type_;
};

////////////////////////////////////////////////////////////////////////////////
// Gridded format
////////////////////////////////////////////////////////////////////////////////
// pxx :: export
// pxx :: instance(["double"])
/** Gridded scattering data field.
 *
 * Holds scattering data in gridded format. The data is in this case given
 * in the form of a rank-7 tensor with the dimensions corresponding to the
 * following grids:
 *     1: Frequency
 *     2: Temperature
 *     3: Incoming azimuth angle
 *     4: Incoming zenith angle
 *     5: Scattering azimuth angle
 *     6: Scattering zenith angle
 */
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
  using VectorPtr = const std::shared_ptr<const eigen::Vector<Scalar>>;
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
  using DataTensorPtr = std::shared_ptr<DataTensor>;

  // pxx :: hide
  /** Create gridded scattering data field.
   * @param f_grid The frequency grid.
   * @param t_grid The temperature grid.
   * @lon_inc The incoming azimuth angle
   * @lat_inc The incoming zenith angle
   * @lon_scat The scattering zenith angle
   * @lat_scat The scattering azimuth angle
   * @data The tensor containing the scattering data.
   */
  ScatteringDataFieldGridded(VectorPtr f_grid,
                             VectorPtr t_grid,
                             VectorPtr lon_inc,
                             VectorPtr lat_inc,
                             VectorPtr lon_scat,
                             VectorPtr lat_scat,
                             DataTensorPtr data)
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

  /** Create gridded scattering data field.
   * @param f_grid The frequency grid.
   * @param t_grid The temperature grid.
   * @lon_inc The incoming azimuth angle
   * @lat_inc The incoming zenith angle
   * @lon_scat The scattering zenith angle
   * @lat_scat The scattering azimuth angle
   * @data The tensor containing the scattering data.
   */
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
        f_grid_map_(f_grid_->data(), n_freqs_),
        t_grid_map_(t_grid_->data(), n_temps_),
        lon_inc_map_(lon_inc_->data(), n_lon_inc_),
        lat_inc_map_(lat_inc_->data(), n_lat_inc_),
        lon_scat_map_(lon_scat_->data(), n_lon_scat_),
        lat_scat_map_(lat_scat_->data(), n_lat_scat_),
        data_(std::make_shared<DataTensor>(data)) {}

  /** Create empty gridded scattering data field.
   *
   * This constructor is useful to pre-allocate data for sequentially
   * loading scattering data from multiple files or that is defined
   * on different grids.
   *
   * @param f_grid The frequency grid.
   * @param t_grid The temperature grid.
   * @lon_inc The incoming azimuth angle
   * @lat_inc The incoming zenith angle
   * @lon_scat The scattering zenith angle
   * @lat_scat The scattering azimuth angle
   * @n_element The number of scattering-data elements, e.g.
   * phase matrix components that need to be stored for this
   * data field.
   */
  ScatteringDataFieldGridded(Vector f_grid,
                             Vector t_grid,
                             Vector &lon_inc,
                             Vector &lat_inc,
                             Vector &lon_scat,
                             Vector &lat_scat,
                             Index n_elements)
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
        f_grid_map_(f_grid_->data(), n_freqs_),
        t_grid_map_(t_grid_->data(), n_temps_),
        lon_inc_map_(lon_inc_->data(), n_lon_inc_),
        lat_inc_map_(lat_inc_->data(), n_lat_inc_),
        lon_scat_map_(lon_scat_->data(), n_lon_scat_),
        lat_scat_map_(lat_scat_->data(), n_lat_scat_),
        data_(std::make_shared<DataTensor>(std::array<Index, 7>{n_freqs_,
                                                                n_temps_,
                                                                n_lon_inc_,
                                                                n_lat_inc_,
                                                                n_lon_scat_,
                                                                n_lat_scat_,
                                                                n_elements})) {}
  /// Shallow copy of the ScatteringDataField.
  ScatteringDataFieldGridded(const ScatteringDataFieldGridded &) = default;

  /// Deep copy of the scattering data.
  ScatteringDataFieldGridded copy() const {
      auto data_new = std::make_shared<DataTensor>(*data_);
      return ScatteringDataFieldGridded(f_grid_,
                                        t_grid_,
                                        lon_inc_,
                                        lat_inc_,
                                        lon_scat_,
                                        lat_scat_,
                                        data_new);
  }

  /** Set scattering data for given frequency and temperature index.
   *
   * This function copies the data from the given scattering data field
   * into the sub-tensor of this objects' data tensor identified by
   * the given frequency and temperature indices. The data is automatically
   * regridded to the scattering data grids of this object.
   *
   * This function is useful to combine scattering data at different
   * temperatures and frequencies that have different scattering grids.
   *
   * @frequency_index The index along the frequency dimension
   * @temperature_index The index along the temperature dimension.
   */
  void set_data(eigen::Index frequency_index,
                eigen::Index temperature_index,
                const ScatteringDataFieldGridded &other) {

      using Regridder = RegularRegridder<Scalar, 2, 3, 4, 5>;

      auto f_grid_other = other.f_grid_;
      auto lon_inc_other = other.lon_inc_;
      auto lat_inc_other = other.lat_inc_;
      auto lon_scat_other = other.lon_scat_;
      auto lat_scat_other = other.lat_scat_;
      auto regridder = Regridder(
          {*lon_inc_other, *lat_inc_other, *lon_scat_other, *lat_scat_other},
          {*lon_inc_, *lat_inc_, *lon_scat_, *lat_scat_});
      auto regridded = regridder.regrid(*other.data_);

      std::array<eigen::Index, 2> data_index = {frequency_index,
                                                temperature_index};
      std::array<eigen::Index, 2> input_index = {0, 0};
      eigen::tensor_index(*data_, data_index) = eigen::tensor_index(regridded,
                                                                    input_index);
  }

  // pxx :: hide
  /** Interpolate data along frequency dimension.
   * @param frequencies The frequency grid to which to interpolate the data
   * @return New scattering data field with the given frequencies as
   * frequency grid.
   */
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
  /** Interpolate data along temperature dimension.
  * @param temperature The temperature grid to which to interpolate the data
  * @return New scattering data field with the given temperatures as
  * temperature grid.
  */
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
      VectorPtr lon_inc_new,
      VectorPtr lat_inc_new,
      VectorPtr lon_scat_new,
      VectorPtr lat_scat_new)
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

  /** Interpolate angular grids.
   * @param lon_inc_new The incoming azimuth angle grid to which to interpolate the data
   * @param lat_inc_new The incoming zenith angle grid to which to interpolate the data
   * @param lon_scat_new The scattering zenith angle grid to which to interpolate the data
   * @param lat_scat_new The scattering azimuth angle grid to which to interpolate the data
   * @return New scattering data field with the given angles as angular grids.
   */
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

  /** Regrid data to new grids.
   * @param f_grid The frequency grid.
   * @param t_grid The temperature grid.
   * @lon_inc The incoming azimuth angle
   * @lat_inc The incoming zenith angle
   * @lon_scat The scattering zenith angle
   * @lat_scat The scattering azimuth angle
   * @return A new ScatteringDataFieldGridded with the given grids.
   */
  // pxx :: hide
  ScatteringDataFieldGridded regrid(VectorPtr f_grid,
                                    VectorPtr t_grid,
                                    VectorPtr lon_inc,
                                    VectorPtr lat_inc,
                                    VectorPtr lon_scat,
                                    VectorPtr lat_scat) const {
      using Regridder = RegularRegridder<Scalar, 0, 1, 2, 3, 4, 5>;
      Regridder regridder({*f_grid_, *t_grid_, *lon_inc_, *lat_inc_, *lon_scat_, *lat_scat_},
                          {*f_grid, *t_grid, *lon_inc, *lat_inc, *lon_scat, *lat_scat});
      auto data_interp = regridder.regrid(*data_);
      auto data_new = std::make_shared<DataTensor>(std::move(data_interp));
      return ScatteringDataFieldGridded(f_grid,
                                        t_grid,
                                        lon_inc,
                                        lat_inc,
                                        lon_scat,
                                        lat_scat,
                                        data_new);
  }

  /** Accumulate scattering data into this object.
   *
   * Regrids the given scattering data field and accumulates its interpolated data tensor
   * into this object's data tensor.
   *
   * @param other The ScatteringDataField to accumulate into this.
   * @return Reference to this object.
   */
  ScatteringDataFieldGridded operator+=(const ScatteringDataFieldGridded &other) {
      auto regridded = other.regrid(f_grid_, t_grid_, lon_inc_, lat_inc_, lon_scat_, lat_scat_);
      *data_ += regridded.get_data();
      return *this;
  }

  ScatteringDataFieldGridded operator+(const ScatteringDataFieldGridded &other) const {
      auto result = copy();
      result += other;
      return result;
  }

  /** In-place scaling scattering data.
   *
   * @param c The scaling factor.
   * @return Reference to this object.
   */
  ScatteringDataFieldGridded operator*=(Scalar c) {
      (*data_) = c * (*data_);
      return *this;
  }

  /** Scale scattering data.
   *
   * @param c The scaling factor.
   * @return A new object containing the scaled scattering data.
   */
  ScatteringDataFieldGridded operator*(Scalar c) const {
      auto result = copy();
      result *= c;
      return result;
  }

  // pxx :: hide
  /** Convert gridded data to spectral format.
   * @param sht SHT instance to use for the transformation.
   * @return The scattering data field transformed to spectral format.
   */
  ScatteringDataFieldSpectral<Scalar> to_spectral(std::shared_ptr<sht::SHT> sht);

  /** Convert gridded data to spectral format.
   * @param l_max The maximum degree l to use in the SH expansion.
   * @param m_max The maximum order m to use in the SH expansion.
   * @return The scattering data field transformed to spectral format.
   */
  ScatteringDataFieldSpectral<Scalar> to_spectral(Index l_max,
                                                  Index m_max) {

      std::shared_ptr<sht::SHT> sht = std::make_shared<sht::SHT>(l_max,
                                                                 m_max,
                                                                 n_lat_scat_,
                                                                 n_lon_scat_);
      return to_spectral(sht);
  }

  /** Convert gridded data to spectral format.
   *
   * This version uses the highest possible values for the maximum order
   * and degree that fulfill the anti-aliasing conditions.
   * @return The scattering data field transformed to spectral format.
   */
  ScatteringDataFieldSpectral<Scalar> to_spectral() {
    Index l_max = ((n_lat_scat_ % 2) == 0) ? n_lat_scat_ - 2 : n_lat_scat_ - 1;
    Index m_max = (n_lon_scat_ > 2) ? (n_lon_scat_ / 2) - 1 : 0;
    return to_spectral(l_max, m_max);
  }

  /// The data tensor containing the scattering data.
  const DataTensor &get_data() const {return *data_;}

 protected:

  VectorPtr f_grid_;
  VectorPtr t_grid_;
  VectorPtr lon_inc_;
  VectorPtr lat_inc_;
  VectorPtr lon_scat_;
  VectorPtr lat_scat_;

  ConstVectorMap f_grid_map_;
  ConstVectorMap t_grid_map_;
  ConstVectorMap lon_inc_map_;
  ConstVectorMap lat_inc_map_;
  ConstVectorMap lon_scat_map_;
  ConstVectorMap lat_scat_map_;

  DataTensorPtr data_;
};

////////////////////////////////////////////////////////////////////////////////
// Spectral format
////////////////////////////////////////////////////////////////////////////////
// pxx :: export
// pxx :: instance(["double"])
/** Scattering data in spectral format.
 *
 * Represents scattering data where the scattering-angle dependency is
 * represented using SH-coefficients.
 */
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
  using VectorPtr = const std::shared_ptr<const eigen::Vector<Scalar>>;
  using ConstVectorMap = eigen::ConstVectorMap<Scalar>;
  using Matrix = eigen::Matrix<Scalar>;
  using MatrixMap = eigen::MatrixMap<Scalar>;
  using ConstMatrixMap = eigen::ConstMatrixMap<Scalar>;
  using ShtPtr = std::shared_ptr<sht::SHT>;

  template <eigen::Index rank>
  using CmplxTensor = eigen::Tensor<std::complex<Scalar>, rank>;
  template <eigen::Index rank>
  using CmplxTensorMap = eigen::TensorMap<std::complex<Scalar>, rank>;
  template <eigen::Index rank>
  using ConstCmplxTensorMap = eigen::ConstTensorMap<std::complex<Scalar>, rank>;
  using DataTensor = eigen::Tensor<std::complex<Scalar>, 6>;
  using DataTensorPtr = std::shared_ptr<DataTensor>;

  // pxx :: hide
  /** Create scattering data field.
   * @param f_grid The frequency grid.
   * @param t_grid The temperature grid.
   * @param lon_inc The longitude grid for the incoming angles.
   * @param lat_inc The latitude grid for the incoming angles.
   * @param sht_scat The SH transform used to expand the scattering-angle
   * dependency.
   * @data The scattering data.
   */
  ScatteringDataFieldSpectral(VectorPtr f_grid,
                              VectorPtr t_grid,
                              VectorPtr lon_inc,
                              VectorPtr lat_inc,
                              ShtPtr sht_scat,
                              DataTensorPtr data)
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

  /** Create scattering data field.
   * @param f_grid The frequency grid.
   * @param t_grid The temperature grid.
   * @param lon_inc The longitude grid for the incoming angles.
   * @param lat_inc The latitude grid for the incoming angles.
   * @param sht_scat The SH transform used to expand the scattering-angle
   * dependency.
   * @data The scattering data.
   */
  ScatteringDataFieldSpectral(const Vector &f_grid,
                              const Vector &t_grid,
                              const Vector &lon_inc,
                              const Vector &lat_inc,
                              const sht::SHT &sht_scat,
                              const DataTensor &data)
      : ScatteringDataFieldBase(f_grid.size(),
                                t_grid.size(),
                                lon_inc.size(),
                                lat_inc.size(),
                                sht_scat.get_n_longitudes(),
                                sht_scat.get_n_latitudes()),
        f_grid_(std::make_shared<Vector>(f_grid)),
        t_grid_(std::make_shared<Vector>(t_grid)),
        lon_inc_(std::make_shared<Vector>(lon_inc)),
        lat_inc_(std::make_shared<Vector>(lat_inc)),
        sht_scat_(std::make_shared<sht::SHT>(sht_scat)),
        f_grid_map_(f_grid_->data(), n_freqs_),
        t_grid_map_(t_grid_->data(), n_temps_),
        lon_inc_map_(lon_inc_->data(), n_freqs_),
        lat_inc_map_(lat_inc_->data(), n_temps_),
        data_(std::make_shared<DataTensor>(data)) {}

  /** Create empty scattering data field.
   * @param f_grid The frequency grid.
   * @param t_grid The temperature grid.
   * @param lon_inc The longitude grid for the incoming angles.
   * @param lat_inc The latitude grid for the incoming angles.
   * @param sht_scat The SH transform used to expand the scattering-angle
   * dependency.
   * @data The scattering data.
   */
  ScatteringDataFieldSpectral(const Vector &f_grid,
                              const Vector &t_grid,
                              const Vector &lon_inc,
                              const Vector &lat_inc,
                              const sht::SHT &sht_scat,
                              Index n_elements)
      : ScatteringDataFieldBase(f_grid.size(),
                                t_grid.size(),
                                lon_inc.size(),
                                lat_inc.size(),
                                sht_scat.get_n_longitudes(),
                                sht_scat.get_n_latitudes()),
        f_grid_(std::make_shared<Vector>(f_grid)),
        t_grid_(std::make_shared<Vector>(t_grid)),
        lon_inc_(std::make_shared<Vector>(lon_inc)),
        lat_inc_(std::make_shared<Vector>(lat_inc)),
        sht_scat_(std::make_shared<sht::SHT>(sht_scat)),
        f_grid_map_(f_grid_->data(), n_freqs_),
        t_grid_map_(t_grid_->data(), n_temps_),
        lon_inc_map_(lon_inc_->data(), n_freqs_),
        lat_inc_map_(lat_inc_->data(), n_temps_),
        data_(std::make_shared<DataTensor>(
            std::array<Index, 6>{n_freqs_,
                                 n_temps_,
                                 n_lon_inc_,
                                 n_lat_inc_,
                                 sht_scat.get_n_spectral_coeffs(),
                                 n_elements})) {}

  /// Shallow copy of the ScatteringDataField.
  ScatteringDataFieldSpectral(const ScatteringDataFieldSpectral &) = default;

  /// Deep copy of the scattering data.
  ScatteringDataFieldSpectral copy() const {
      auto data_new = std::make_shared<DataTensor>(*data_);
      return ScatteringDataFieldSpectral(f_grid_,
                                        t_grid_,
                                        lon_inc_,
                                        lat_inc_,
                                        sht_scat_,
                                        data_new);
  }

  /** Set scattering data for given frequency and temperature index.
   *
   * This function copies the data from the given scattering data field
   * into the sub-tensor of this objects' data tensor identified by
   * the given frequency and temperature indices. The data is automatically
   * regridded to the scattering data grids of this object.
   *
   * This function is useful to combine scattering data at different
   * temperatures and frequencies that have different scattering grids.
   *
   * @frequency_index The index along the frequency dimension
   * @temperature_index The index along the temperature dimension.
   */
  void set_data(eigen::Index frequency_index,
                eigen::Index temperature_index,
                const ScatteringDataFieldSpectral &other) {

      using Regridder = RegularRegridder<Scalar, 2, 3>;

      auto lon_inc_other = other.lon_inc_;
      auto lat_inc_other = other.lat_inc_;
      auto regridder = Regridder(
          {*lon_inc_other, *lat_inc_other},
          {*lon_inc_, *lat_inc_});
      auto regridded = regridder.regrid(*other.data_);

      std::array<eigen::Index, 2> data_index = {frequency_index,
                                                temperature_index};
      std::array<eigen::Index, 2> input_index = {0, 0};

      eigen::IndexArray<3> dimensions_loop = {n_lon_inc_,
                                              n_lat_inc_,
                                              data_->dimension(5)};
      auto data_map = eigen::tensor_index(*data_, data_index);
      auto other_data_map = eigen::tensor_index(regridded, input_index);
      for (eigen::DimensionCounter<3> i{dimensions_loop}; i; ++i) {
          auto result = eigen::get_subvector<2>(data_map, i.coordinates);
          auto in_l = eigen::get_subvector<2>(data_map, i.coordinates);
          auto in_r = eigen::get_subvector<2>(other_data_map, i.coordinates);
          result = sht::SHT::add_coeffs(*sht_scat_, in_l, *other.sht_scat_, in_r);
      }
  }

  // pxx :: hide
  /** Interpolate data along frequency dimension.
   * @param frequencies The frequencies to which to interpolate the data.
   * @return New ScatteringDataFieldSpectral with the data interpolated
   * to the given frequencies.
   */
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
  /** Interpolate data along temperature dimension.
   * @param temperatures The temperatures to which to interpolate the data.
   * @return New ScatteringDataFieldSpectral with the data interpolated
   * to the given temperatures.
   */
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
  /** Interpolate data along incoming angles.
   * @param temperatures The temperatures to which to interpolate the data.
   * @return New ScatteringDataFieldSpectral with the data interpolated
   * to the given temperatures.
   */
  ScatteringDataFieldSpectral interpolate_angles(
      VectorPtr lon_inc_new,
      VectorPtr lat_inc_new) const {
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

  /** Regrid data to new grids.
   * @param f_grid The frequency grid.
   * @param t_grid The temperature grid.
   * @lon_inc The incoming azimuth angle
   * @lat_inc The incoming zenith angle
   * @return A new ScatteringDataFieldSpectral with the given grids.
   */
  // pxx :: hide
  ScatteringDataFieldSpectral regrid(VectorPtr f_grid,
                                     VectorPtr t_grid,
                                     VectorPtr lon_inc,
                                     VectorPtr lat_inc) const {
    using Regridder = RegularRegridder<Scalar, 0, 1, 2, 3>;
    Regridder regridder({*f_grid_, *t_grid_, *lon_inc_, *lat_inc_},
                        {*f_grid, *t_grid, *lon_inc, *lat_inc});
    auto data_interp = regridder.regrid(*data_);
    auto data_new = std::make_shared<DataTensor>(std::move(data_interp));
    return ScatteringDataFieldSpectral(f_grid,
                                       t_grid,
                                       lon_inc,
                                       lat_inc,
                                       sht_scat_,
                                       data_new);
  }

  /** Accumulate scattering data into this object.
   *
   * Regrids the given scattering data field and accumulates its interpolated data tensor
   * into this object's data tensor.
   *
   * @param other The ScatteringDataField to accumulate into this.
   * @return Reference to this object.
   */
  ScatteringDataFieldSpectral& operator+=(const ScatteringDataFieldSpectral &other) {
      auto regridded = other.regrid(f_grid_, t_grid_, lon_inc_, lat_inc_);
      eigen::IndexArray<5> dimensions_loop = {n_freqs_,
                                              n_temps_,
                                              n_lon_inc_,
                                              n_lat_inc_,
                                              data_->dimension(5)};
      for (eigen::DimensionCounter<5> i{dimensions_loop}; i; ++i) {
          auto result = eigen::get_subvector<4>(*data_, i.coordinates);
          auto in_l = eigen::get_subvector<4>(*data_, i.coordinates);
          auto in_r = eigen::get_subvector<4>(*regridded.data_, i.coordinates);
          result = sht::SHT::add_coeffs(*sht_scat_, in_l, *regridded.sht_scat_, in_r);
      }
      return *this;
  }

  /** Add scattering data fields.
   *
   * Regrids the given scattering data field to the grids of this object and computes
   * the sum of the two scattering data fields.
   *
   * @param other The ScatteringDataField to accumulate into this.
   * @return Reference to this object.
   */
  ScatteringDataFieldSpectral operator+(const ScatteringDataFieldSpectral &other) {
      auto result = copy();
      result += other;
      return result;
  }

  /** In-place scaling scattering data.
   *
   * @param c The scaling factor.
   * @return Reference to this object.
   */
  ScatteringDataFieldSpectral & operator*=(Scalar c) {
      (*data_) = c * (*data_) ;
      return *this;
  }

  /** Scale scattering data.
   *
   * @param c The scaling factor.
   * @return A new object containing the scaled scattering data.
   */
  ScatteringDataFieldSpectral operator*(Scalar c) const {
      auto result = copy();
      result *= c;
      return result;
  }

  ScatteringDataFieldGridded<Scalar> to_gridded();
  ScatteringDataFieldFullySpectral<Scalar> to_fully_spectral(ShtPtr sht);
  ScatteringDataFieldFullySpectral<Scalar> to_fully_spectral(Index l_max,
                                                             Index m_max) {
    std::shared_ptr<sht::SHT> sht =
        std::make_shared<sht::SHT>(l_max, m_max, n_lat_inc_, n_lon_inc_);
    return to_fully_spectral(sht);
  }
  ScatteringDataFieldFullySpectral<Scalar> to_fully_spectral() {
      Index l_max = ((n_lat_inc_ % 2) == 0) ? n_lat_inc_ - 2 : n_lat_inc_ - 1;
      Index m_max = (n_lon_inc_ > 2) ? (n_lon_inc_ / 2) - 1 : 0;
      return to_fully_spectral(l_max, m_max);
  }

  const DataTensor &get_data() const {return *data_;}

  sht::SHT &get_sht_scat() const {return *sht_scat_;}

 protected:

  VectorPtr f_grid_;
  VectorPtr t_grid_;
  VectorPtr lon_inc_;
  VectorPtr lat_inc_;
  ShtPtr sht_scat_;

  ConstVectorMap f_grid_map_;
  ConstVectorMap t_grid_map_;
  ConstVectorMap lon_inc_map_;
  ConstVectorMap lat_inc_map_;

  DataTensorPtr data_;
};

// pxx :: export
// pxx :: instance(["double"])
////////////////////////////////////////////////////////////////////////////////
// Fully-spectral format
////////////////////////////////////////////////////////////////////////////////
/** Fully-spectral scattering data.
 *
 * Represents scattering data fields whose dependency to both the incoming and
 * the scattering angles is represented using SHs.
 *
 */
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
  using VectorPtr = const std::shared_ptr<const eigen::Vector<Scalar>>;
  using ConstVectorMap = eigen::ConstVectorMap<Scalar>;
  using Matrix = eigen::Matrix<Scalar>;
  using MatrixMap = eigen::MatrixMap<Scalar>;
  using ConstMatrixMap = eigen::ConstMatrixMap<Scalar>;
  using ShtPtr = std::shared_ptr<sht::SHT>;

  template <eigen::Index rank>
  using CmplxTensor = eigen::Tensor<std::complex<Scalar>, rank>;
  template <eigen::Index rank>
  using CmplxTensorMap = eigen::TensorMap<std::complex<Scalar>, rank>;
  template <eigen::Index rank>
  using ConstCmplxTensorMap = eigen::ConstTensorMap<std::complex<Scalar>, rank>;
  using DataTensor = eigen::Tensor<std::complex<Scalar>, 5>;
  using DataTensorPtr = std::shared_ptr<DataTensor>;

  // pxx :: hide
  /** Create scattering data field.
   * @param f_grid The frequency grid.
   * @param t_grid The temperature grid.
   * @param sht_inc The SH transform used to expand the incoming-angle
   * dependency.
   * @param sht_scat The SH transform used to expand the scattering-angle
   * dependency.
   * @data The scattering data.
   */
  ScatteringDataFieldFullySpectral(VectorPtr f_grid,
                                   VectorPtr t_grid,
                                   ShtPtr sht_inc,
                                   ShtPtr sht_scat,
                                   DataTensorPtr data)
      : ScatteringDataFieldBase(f_grid->size(),
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

  /** Create scattering data field.
   * @param f_grid The frequency grid.
   * @param t_grid The temperature grid.
   * @param sht_inc The SH transform used to expand the incoming-angle
   * dependency.
   * @param sht_scat The SH transform used to expand the scattering-angle
   * dependency.
   * @data The scattering data.
   */
  ScatteringDataFieldFullySpectral(const Vector &f_grid,
                                   const Vector &t_grid,
                                   const sht::SHT &sht_inc,
                                   const sht::SHT &sht_scat,
                                   const DataTensor &data)
      : ScatteringDataFieldBase(f_grid.size(),
                                t_grid.size(),
                                sht_inc.get_n_longitudes(),
                                sht_inc.get_n_latitudes(),
                                sht_scat.get_n_longitudes(),
                                sht_scat.get_n_latitudes()),
        f_grid_(std::make_shared<Vector>(f_grid)),
        t_grid_(std::make_shared<Vector>(t_grid)),
        sht_inc_(std::make_shared<sht::SHT>(sht_inc)),
        sht_scat_(std::make_shared<sht::SHT>(sht_scat)),
        f_grid_map_(f_grid_->data(), n_freqs_),
        t_grid_map_(t_grid_->data(), n_temps_),
        data_(std::make_shared<DataTensor>(data)) {}

  /** Create empty scattering data field.
   * @param f_grid The frequency grid.
   * @param t_grid The temperature grid.
   * @param sht_inc The SH transform used to expand the incoming-angle
   * dependency.
   * @param sht_scat The SH transform used to expand the scattering-angle
   * dependency.
   * @data The scattering data.
   */
  ScatteringDataFieldFullySpectral(const Vector &f_grid,
                                   const Vector &t_grid,
                                   const sht::SHT &sht_inc,
                                   const sht::SHT &sht_scat,
                                   Index n_elements)
      : ScatteringDataFieldBase(f_grid.size(),
                                t_grid.size(),
                                sht_inc.get_n_longitudes(),
                                sht_inc.get_n_latitudes(),
                                sht_scat.get_n_longitudes(),
                                sht_scat.get_n_latitudes()),
        f_grid_(std::make_shared<Vector>(f_grid)),
        t_grid_(std::make_shared<Vector>(t_grid)),
        sht_inc_(std::make_shared<sht::SHT>(sht_inc)),
        sht_scat_(std::make_shared<sht::SHT>(sht_scat)),
        f_grid_map_(f_grid_->data(), n_freqs_),
        t_grid_map_(t_grid_->data(), n_temps_),
        data_(std::make_shared<DataTensor>(
            std::array<Index, 5>{f_grid.size(),
                                 t_grid.size(),
                                 sht_inc.get_n_spectral_coeffs_cmplx(),
                                 sht_scat.get_n_spectral_coeffs(),
                                 n_elements})) {}

  /// Shallow copy of the ScatteringDataField.
  ScatteringDataFieldFullySpectral(const ScatteringDataFieldFullySpectral &) = default;

  /// Deep copy of the scattering data.
  ScatteringDataFieldFullySpectral copy() const {
    auto data_new = std::make_shared<DataTensor>(*data_);
    return ScatteringDataFieldFullySpectral(f_grid_,
                                            t_grid_,
                                            sht_inc_,
                                            sht_scat_,
                                            data_new);
  }


  /** Set scattering data for given frequency and temperature index.
   *
   * This function copies the data from the given scattering data field
   * into the sub-tensor of this objects' data tensor identified by
   * the given frequency and temperature indices. The data is automatically
   * regridded to the scattering data grids of this object.
   *
   * This function is useful to combine scattering data at different
   * temperatures and frequencies that have different scattering grids.
   *
   * @frequency_index The index along the frequency dimension
   * @temperature_index The index along the temperature dimension.
   */
  void set_data(eigen::Index frequency_index,
                eigen::Index temperature_index,
                const ScatteringDataFieldFullySpectral &other) {

      using Regridder = RegularRegridder<Scalar, 2, 3>;

      std::array<eigen::Index, 2> data_index = {frequency_index,
                                                temperature_index};
      std::array<eigen::Index, 2> input_index = {0, 0};
      auto data_map = eigen::tensor_index(*data_, data_index);
      auto other_data_map = eigen::tensor_index(*other.data_, input_index);

      eigen::IndexArray<1> dimensions_loop = {data_->dimension(5)};
      for (eigen::DimensionCounter<1> i{dimensions_loop}; i; ++i) {
          auto result = eigen::get_submatrix<0, 1>(data_map, i.coordinates);
          auto in_l = eigen::get_submatrix<0, 1>(data_map, i.coordinates);
          auto in_r = eigen::get_submatrix<0, 1>(other_data_map, i.coordinates);
          result = sht::SHT::add_coeffs(*sht_inc_, *sht_scat_, in_l,
                                        *other.sht_inc_, *other.sht_scat_, in_r);
      }
  }

  // pxx :: hide
  /** Interpolate data along frequency dimension.
   * @param frequencies The frequencies to which to interpolate the data.
   * @return New ScatteringDataFieldFullySpectral with the data interpolated
   * to the given frequencies.
   */
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
  /** Interpolate data along temperature dimension.
   * @param temperatures The temperatures to which to interpolate the data.
   * @return New ScatteringDataFieldFullySpectral with the data interpolated
   * to the given temperatures.
   */
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

  /** Regrid data to new grids.
   * @param f_grid The frequency grid.
   * @param t_grid The temperature grid.
   * @return A new ScatteringDataFieldFullySpectral with the given grids.
   */
  // pxx :: hide
  ScatteringDataFieldFullySpectral regrid(VectorPtr f_grid,
                                          VectorPtr t_grid) const {
      using Regridder = RegularRegridder<Scalar, 0, 1>;
      Regridder regridder({*f_grid_, *t_grid_},
                          {*f_grid, *t_grid});
      auto data_interp = regridder.regrid(*data_);
      auto data_new = std::make_shared<DataTensor>(std::move(data_interp));
      return ScatteringDataFieldFullySpectral(f_grid,
                                              t_grid,
                                              sht_inc_,
                                              sht_scat_,
                                              data_new);
  }


  /** Accumulate scattering data into this object.
   *
   * Regrids the given scattering data field and accumulates its interpolated data tensor
   * into this object's data tensor.
   *
   * @param other The ScatteringDataField to accumulate into this.
   * @return Reference to this object.
   */
  ScatteringDataFieldFullySpectral& operator+=(const ScatteringDataFieldFullySpectral &other) {
      auto regridded = other.regrid(f_grid_, t_grid_);
      eigen::IndexArray<3> dimensions_loop = {n_freqs_,
                                              n_temps_,
                                              data_->dimension(4)};
      for (eigen::DimensionCounter<3> i{dimensions_loop}; i; ++i) {
          auto result = eigen::get_submatrix<2, 3>(*data_, i.coordinates);
          auto in_l = eigen::get_submatrix<2, 3>(*data_, i.coordinates);
          auto in_r = eigen::get_submatrix<2, 3>(*regridded.data_, i.coordinates);
          result = sht::SHT::add_coeffs(*sht_inc_, *sht_scat_, in_l,
                                        *regridded.sht_inc_, *regridded.sht_scat_, in_r);
      }
      return *this;
  }

  /** Add scattering data fields.
   *
   * Regrids the given scattering data field to the grids of this object and computes
   * the sum of the two scattering data fields.
   *
   * @param other The other scattering data field to add to this.
   * @return A new ScatteringDataFieldFullySpectral object representing the sum of the
   * two other fields.
   */
  ScatteringDataFieldFullySpectral operator+(const ScatteringDataFieldFullySpectral &other) const {
      auto result = copy();
      result += other;
      return result;
  }

  /** In-place scaling scattering data.
   *
   * @param c The scaling factor.
   * @return Reference to this object.
   */
  ScatteringDataFieldFullySpectral & operator*=(Scalar c) {
      *data_ = c * (*data_) ;
      return *this;
  }

  /** Scale scattering data.
   *
   * @param c The scaling factor.
   * @return A new object containing the scaled scattering data.
   */
  ScatteringDataFieldFullySpectral operator*(Scalar c) const {
      auto result = copy();
      result *= c;
      return result;
  }

  sht::SHT &get_sht_scat() const {return *sht_scat_;}
  sht::SHT &get_sht_inc() const {return *sht_inc_;}

  ScatteringDataFieldSpectral<Scalar> to_spectral();

  const DataTensor &get_data() const {return *data_;}

 protected:

  VectorPtr f_grid_;
  VectorPtr t_grid_;
  VectorPtr lon_inc_;
  VectorPtr lat_inc_;
  ShtPtr sht_inc_;
  ShtPtr sht_scat_;

  ConstVectorMap f_grid_map_;
  ConstVectorMap t_grid_map_;

  DataTensorPtr data_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation of conversion methods.
////////////////////////////////////////////////////////////////////////////////

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
        sht_inc_->synthesize_cmplx(eigen::get_subvector<2>(*data_, i.coordinates));
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

}  // namespace scatlib

#endif
