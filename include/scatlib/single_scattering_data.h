/** \file scattering_data.h
 *
 * Provides the SingleScatteringData class representing the single scattering
 * data of a single scattering particle.
 *
 * @author Simon Pfreundschuh, 2020
 */
#ifndef __SCATLIB_SINGLE_SCATTERING_DATA__
#define __SCATLIB_SINGLE_SCATTERING_DATA__

#include <scatlib/eigen.h>
#include <scatlib/sht.h>
#include <scatlib/interpolation.h>
#include <scatlib/scattering_data_field.h>
#include <memory>
#include <cassert>

namespace scatlib {

////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
class SingleScatteringDataGridded;
template <typename Scalar>
class SingleScatteringDataSpectral;
template <typename Scalar>
class SingleScatteringDataFullySpectral;

////////////////////////////////////////////////////////////////////////////////
// Interface for format-specific implementations.
////////////////////////////////////////////////////////////////////////////////

// pxx :: export
enum class ParticleType {
  Spherical = 0,
  TotallyRandom = 1,
  AzimuthallyRandom = 2,
  General = 3
};

namespace detail {

Index get_n_phase_matrix_elements(ParticleType type) {
  switch (type) {
    case ParticleType::Spherical:
      return 1;
    case ParticleType::TotallyRandom:
      return 6;
    case ParticleType::AzimuthallyRandom:
      return 16;
    case ParticleType::General:
      return 16;
  }
}

Index get_n_extinction_matrix_elements(ParticleType type) {
  switch (type) {
    case ParticleType::Spherical:
      return 1;
    case ParticleType::TotallyRandom:
      return 1;
    case ParticleType::AzimuthallyRandom:
      return 3;
    case ParticleType::General:
      return 4;
  }
}

Index get_n_absorption_vector_elements(ParticleType type) {
    switch (type) {
    case ParticleType::Spherical:
        return 1;
    case ParticleType::TotallyRandom:
        return 1;
    case ParticleType::AzimuthallyRandom:
        return 2;
    case ParticleType::General:
        return 4;
    }
}

template <typename T>
struct ConditionalDeleter {
  void operator()(T *ptr) {
    if (ptr && do_delete) {
      delete ptr;
    }
  }

  bool do_delete;
};

template <typename T>
using ConversionPtr = std::unique_ptr<T, ConditionalDeleter<T>>;

template <typename T>
ConversionPtr<T> make_conversion_ptr(T *t, bool do_delete) {
  return ConversionPtr<T>(t, ConditionalDeleter<T>{do_delete});
}

}  // namespace detail

class SingleScatteringDataImpl {
 public:


  virtual void set_data(Index f_index, Index t_index, const SingleScatteringDataImpl &) = 0;


  // Interpolation functions
  virtual SingleScatteringDataImpl *interpolate_frequency(
      eigen::VectorPtr<double> frequencies) = 0;
  virtual SingleScatteringDataImpl *interpolate_temperature(
      eigen::VectorPtr<double> temperatures) = 0;
  virtual SingleScatteringDataImpl *interpolate_angles(
      eigen::VectorPtr<double> lon_inc,
      eigen::VectorPtr<double> lat_inc,
      eigen::VectorPtr<double> lon_scat,
      eigen::VectorPtr<double> lat_scat) = 0;

  virtual eigen::Vector<double> get_f_grid() = 0;
  virtual eigen::Vector<double> get_t_grid() = 0;

  // Data access.
  virtual eigen::Tensor<double, 5> get_phase_matrix() = 0;
  virtual eigen::Tensor<double, 5> get_extinction_matrix() = 0;
  virtual eigen::Tensor<double, 5> get_absorption_vector() = 0;

  // Addition
  virtual void operator+=(const SingleScatteringDataImpl *other) = 0;
  virtual SingleScatteringDataImpl *operator+(
      const SingleScatteringDataImpl *other) = 0;

  virtual detail::ConversionPtr<const SingleScatteringDataGridded<double>> to_gridded() const = 0;

  // Conversion operators
  // virtual operator SingleScatteringDataGridded<float>() = 0;
  //virtual operator SingleScatteringDataGridded<double>() = 0;
  // template <typename Scalar>
  // virtual operator SingleScatteringDataSpectral<Scalar>() = 0;
  // template <typename Scalar>
  // virtual operator SingleScatteringDataFullySpectral<Scalar>() = 0;
};

////////////////////////////////////////////////////////////////////////////////
// Format-agnostic single scattering data class
////////////////////////////////////////////////////////////////////////////////

// pxx :: export
/** Format-agnostic interface and container for single-scattering data.
 *
 * This class provides a format-agnostic interface and container for single
 * scattering data. This means that it can store scattering data in any format
 * and allows manipulating and combining this data with any other single
 * scattering data in any other format.
 */
class SingleScatteringData {
 public:

    /** Create from existing pointer to implementation object.
     * @param data Pointer to existing format-specific scattering data object.
     */
    // pxx :: hide
  SingleScatteringData(SingleScatteringDataImpl *data)
      : data_(data) {}

  /** Create from gridded scattering data.
   *
   * This constructor creates a single scattering data object, whose data is
   * stored in gridded format.
   *
   * @param f_grid The frequency grid.
   * @param t_grid The temperature grid.
   * @param lon_inc The incoming-angle longitude grid.
   * @param lat_inc The incoming-angle latitude grid.
   * @param lon_scat The scattering-angle longitude grid.
   * @param lat_scat The scattering-angle latitude grid.
   * @param phase_matrix Tensor containing the phase matrix data.
   * @param extinction_matrix Tensor containing the extinction matrix data.
   * @param absorption_vector Tensor containing the absorption vector data.
   * @param backscattering_coeff Tensor containing the backscattering
   * coefficients
   * @param forwardscattering_coeff Tensor containing the forwardscattering
   * coefficients.
   */
    // pxx :: hide
    SingleScatteringData(scatlib::eigen::VectorPtr<double> f_grid,
                         scatlib::eigen::VectorPtr<double> t_grid,
                         scatlib::eigen::VectorPtr<double> lon_inc,
                         scatlib::eigen::VectorPtr<double> lat_inc,
                         scatlib::eigen::VectorPtr<double> lon_scat,
                         scatlib::eigen::VectorPtr<double> lat_scat,
                         scatlib::eigen::TensorPtr<double, 7> phase_matrix,
                         scatlib::eigen::TensorPtr<double, 7> extinction_matrix,
                         scatlib::eigen::TensorPtr<double, 7> absorption_vector,
                         scatlib::eigen::TensorPtr<double, 4> backscattering_coeff,
                         scatlib::eigen::TensorPtr<double, 4> forwardscattering_coeff);

    SingleScatteringData(
        scatlib::eigen::Vector<double> f_grid,
        scatlib::eigen::Vector<double> t_grid,
        scatlib::eigen::Vector<double> lon_inc,
        scatlib::eigen::Vector<double> lat_inc,
        scatlib::eigen::Vector<double> lon_scat,
        scatlib::eigen::Vector<double> lat_scat,
        scatlib::eigen::Tensor<double, 7> phase_matrix,
        scatlib::eigen::Tensor<double, 7> extinction_matrix,
        scatlib::eigen::Tensor<double, 7> absorption_vector,
        scatlib::eigen::Tensor<double, 4> backscattering_coeff,
        scatlib::eigen::Tensor<double, 4> forwardscattering_coeff)
        : SingleScatteringData(
              std::make_shared<eigen::Vector<double>>(f_grid),
              std::make_shared<eigen::Vector<double>>(t_grid),
              std::make_shared<eigen::Vector<double>>(lon_inc),
              std::make_shared<eigen::Vector<double>>(lat_inc),
              std::make_shared<eigen::Vector<double>>(lon_scat),
              std::make_shared<eigen::Vector<double>>(lat_scat),
              std::make_shared<eigen::Tensor<double, 7>>(phase_matrix),
              std::make_shared<eigen::Tensor<double, 7>>(extinction_matrix),
              std::make_shared<eigen::Tensor<double, 7>>(absorption_vector),
              std::make_shared<eigen::Tensor<double, 4>>(backscattering_coeff),
              std::make_shared<eigen::Tensor<double, 4>>(
                  forwardscattering_coeff)) {}

    SingleScatteringData(scatlib::eigen::Vector<double> f_grid,
                         scatlib::eigen::Vector<double> t_grid,
                         scatlib::eigen::Vector<double> lon_inc,
                         scatlib::eigen::Vector<double> lat_inc,
                         scatlib::eigen::Vector<double> lon_scat,
                         scatlib::eigen::Vector<double> lat_scat,
                         ParticleType type)
        : SingleScatteringData(
              std::make_shared<eigen::Vector<double>>(f_grid),
              std::make_shared<eigen::Vector<double>>(t_grid),
              std::make_shared<eigen::Vector<double>>(lon_inc),
              std::make_shared<eigen::Vector<double>>(lat_inc),
              std::make_shared<eigen::Vector<double>>(lon_scat),
              std::make_shared<eigen::Vector<double>>(lat_scat),
              std::make_shared<eigen::Tensor<double, 7>>(std::array<Index, 7>{
                  f_grid.size(),
                  t_grid.size(),
                  lon_inc.size(),
                  lat_inc.size(),
                  lon_scat.size(),
                  lat_scat.size(),
                  detail::get_n_phase_matrix_elements(type)}),
              std::make_shared<eigen::Tensor<double, 7>>(std::array<Index, 7>{
                  f_grid.size(),
                  t_grid.size(),
                  lon_inc.size(),
                  lat_inc.size(),
                  1,
                  1,
                  detail::get_n_extinction_matrix_elements(type)}),
              std::make_shared<eigen::Tensor<double, 7>>(std::array<Index, 7>{
                  f_grid.size(),
                  t_grid.size(),
                  lon_inc.size(),
                  lat_inc.size(),
                  1,
                  1,
                  detail::get_n_absorption_vector_elements(type)}),
              std::make_shared<eigen::Tensor<double, 4>>(
                  std::array<Index, 4>{f_grid.size(),
                                       t_grid.size(),
                                       lon_inc.size(),
                                       lat_inc.size()}),
              std::make_shared<eigen::Tensor<double, 4>>(
                  std::array<Index, 4>{f_grid.size(),
                                       t_grid.size(),
                                       lon_inc.size(),
                                       lat_inc.size()})) {}

    eigen::Vector<double> get_f_grid() { return data_->get_f_grid(); }
    eigen::Vector<double> get_t_grid() { return data_->get_t_grid(); }

    void set_data(Index f_index,
                  Index t_index,
                  const SingleScatteringData &other) {
      data_->set_data(f_index, t_index, *other.data_);
  }

  // Interpolation functions

  SingleScatteringData interpolate_frequency(eigen::Vector<double> frequencies) {
    auto result = data_->interpolate_frequency(
        std::make_shared<eigen::Vector<double>>(frequencies));
    return SingleScatteringData(std::move(result));
  }

  SingleScatteringData interpolate_temperature(eigen::Vector<double> temperatures) {
    auto result = data_->interpolate_temperature(
        std::make_shared<eigen::Vector<double>>(temperatures));
    return SingleScatteringData(std::move(result));
  }

  SingleScatteringData interpolate_angles(eigen::Vector<double> lon_inc,
                                          eigen::Vector<double> lat_inc,
                                          eigen::Vector<double> lon_scat,
                                          eigen::Vector<double> lat_scat) {
    auto result = data_->interpolate_angles(
        std::make_shared<eigen::Vector<double>>(lon_inc),
        std::make_shared<eigen::Vector<double>>(lat_inc),
        std::make_shared<eigen::Vector<double>>(lon_scat),
        std::make_shared<eigen::Vector<double>>(lat_scat));
    return SingleScatteringData(std::move(result));
  }

  // Data access.
  eigen::Tensor<double, 5> get_phase_matrix() {
    return data_->get_phase_matrix(); }

  eigen::Tensor<double, 5> get_extinction_matrix() {
    return data_->get_extinction_matrix();
  }

  eigen::Tensor<double, 5> get_absorption_vector() {
    return data_->get_absorption_vector();
  }

  // Addition
  SingleScatteringData &operator+=(const SingleScatteringData &other) {
    data_->operator+=(other.data_);
    return *this;
  }
  SingleScatteringData operator+(const SingleScatteringData &other) {
    return SingleScatteringData(data_->operator+(other.data_));
  }

private:
  SingleScatteringDataImpl *data_;
};

////////////////////////////////////////////////////////////////////////////////
// Format-specific implementations
////////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
class SingleScatteringDataBase {
 public:
  using VectorPtr = std::shared_ptr<eigen::Vector<Scalar>>;
  using ScatteringCoeffPtr = std::shared_ptr<eigen::Tensor<Scalar, 4>>;

  SingleScatteringDataBase(eigen::VectorPtr<Scalar> f_grid,
                           eigen::VectorPtr<Scalar> t_grid,
                           eigen::TensorPtr<Scalar, 4> backscattering_coeff,
                           eigen::TensorPtr<Scalar, 4> forwardscattering_coeff)
      : n_freqs_(f_grid->size()),
        n_temps_(t_grid->size()),
        f_grid_(f_grid),
        t_grid_(t_grid),
        backscattering_coeff_(backscattering_coeff),
        forwardscattering_coeff_(forwardscattering_coeff) {}

  void operator+=(const SingleScatteringDataBase &other) {
    backscattering_coeff_->operator+=(*other.backscattering_coeff_);
    forwardscattering_coeff_->operator+=(*other.forwardscattering_coeff_);
  }

 protected:
  size_t n_freqs_, n_temps_;
  VectorPtr f_grid_;
  VectorPtr t_grid_;
  ScatteringCoeffPtr backscattering_coeff_;
  ScatteringCoeffPtr forwardscattering_coeff_;
};

template <typename Scalar>
class SingleScatteringDataGridded : public SingleScatteringDataBase<Scalar>,
                                    public SingleScatteringDataImpl {
  using SingleScatteringDataBase<Scalar>::backscattering_coeff_;
  using SingleScatteringDataBase<Scalar>::f_grid_;
  using SingleScatteringDataBase<Scalar>::forwardscattering_coeff_;
  using SingleScatteringDataBase<Scalar>::n_freqs_;
  using SingleScatteringDataBase<Scalar>::n_temps_;
  using SingleScatteringDataBase<Scalar>::t_grid_;

 public:
  using Vector = eigen::Vector<Scalar>;
  using VectorPtr = std::shared_ptr<Vector>;
  using DataTensor = eigen::Tensor<Scalar, 7>;
  using DataPtr = std::shared_ptr<DataTensor>;
  using ScatteringCoeff = eigen::Tensor<Scalar, 4>;
  using ScatteringCoeffPtr = std::shared_ptr<ScatteringCoeff>;
  using OtherScalar =
      std::conditional<std::is_same<Scalar, double>::value, float, double>;

  SingleScatteringDataGridded(
      VectorPtr f_grid,
      VectorPtr t_grid,
      ScatteringDataFieldGridded<Scalar> phase_matrix,
      ScatteringDataFieldGridded<Scalar> extinction_matrix,
      ScatteringDataFieldGridded<Scalar> absorption_vector,
      ScatteringCoeffPtr backscattering_coeff,
      ScatteringCoeffPtr forwardscattering_coeff)

      : SingleScatteringDataBase<Scalar>(f_grid,
                                         t_grid,
                                         backscattering_coeff,
                                         forwardscattering_coeff),
        phase_matrix_(phase_matrix),
        extinction_matrix_(extinction_matrix),
        absorption_vector_(absorption_vector) {}

  SingleScatteringDataGridded(VectorPtr f_grid,
                              VectorPtr t_grid,
                              VectorPtr lon_inc,
                              VectorPtr lat_inc,
                              VectorPtr lon_scat,
                              VectorPtr lat_scat,
                              DataPtr phase_matrix,
                              DataPtr extinction_matrix,
                              DataPtr absorption_vector,
                              ScatteringCoeffPtr backscattering_coeff,
                              ScatteringCoeffPtr forwardscattering_coeff)
      : SingleScatteringDataBase<Scalar>(f_grid,
                                         t_grid,
                                         backscattering_coeff,
                                         forwardscattering_coeff),
        phase_matrix_{f_grid,
                      t_grid,
                      lon_inc,
                      lat_inc,
                      lon_scat,
                      lat_scat,
                      phase_matrix},
        extinction_matrix_{f_grid,
                           t_grid,
                           lon_inc,
                           lat_inc,
                dummy_grid_,
                dummy_grid_,
                           extinction_matrix},
        absorption_vector_{f_grid,
                           t_grid,
                           lon_inc,
                           lat_inc,
                dummy_grid_,
                dummy_grid_,
                           absorption_vector} {}

  eigen::Vector<double> get_f_grid() { return *f_grid_; }
  eigen::Vector<double> get_t_grid() { return *t_grid_; }

  void set_data(Index f_index,
                Index t_index,
                const SingleScatteringDataImpl &other) {
    auto converted = other.to_gridded();
    phase_matrix_.set_data(f_index, t_index, converted->phase_matrix_);
    extinction_matrix_.set_data(f_index,
                                t_index,
                                converted->extinction_matrix_);
    absorption_vector_.set_data(f_index,
                                t_index,
                                converted->absorption_vector_);
  }

  eigen::Tensor<Scalar, 5> get_phase_matrix() {
    return eigen::tensor_index<2>(phase_matrix_.get_data(), {0, 0});
  }

  eigen::Tensor<Scalar, 5> get_extinction_matrix() {
    return eigen::tensor_index<2>(extinction_matrix_.get_data(), {0, 0});
  }

  eigen::Tensor<Scalar, 5> get_absorption_vector() {
    return eigen::tensor_index<2>(absorption_vector_.get_data(), {0, 0});
  }

  SingleScatteringDataGridded copy() {
    return SingleScatteringDataGridded(
        std::make_shared<Vector>(*f_grid_),
        std::make_shared<Vector>(*t_grid_),
        phase_matrix_.copy(),
        extinction_matrix_.copy(),
        absorption_vector_.copy(),
        std::make_shared<ScatteringCoeff>(*backscattering_coeff_),
        std::make_shared<ScatteringCoeff>(*forwardscattering_coeff_));
  }

  SingleScatteringDataImpl *interpolate_frequency(
      eigen::VectorPtr<double> frequencies) {
    auto phase_matrix = ScatteringDataFieldGridded<Scalar>(
        phase_matrix_.interpolate_frequency(frequencies));
    auto extinction_matrix = ScatteringDataFieldGridded<Scalar>(
        extinction_matrix_.interpolate_frequency(frequencies));
    auto absorption_vector = ScatteringDataFieldGridded<Scalar>(
        absorption_vector_.interpolate_frequency(frequencies));

    using Regridder = RegularRegridder<Scalar, 0>;
    Regridder scat_coeff_regridder({*f_grid_}, {*frequencies});
    auto backscattering_coeff = std::make_shared<ScatteringCoeff>(
        scat_coeff_regridder.regrid(*backscattering_coeff_));
    auto forwardscattering_coeff = std::make_shared<ScatteringCoeff>(
        scat_coeff_regridder.regrid(*forwardscattering_coeff_));
    return new SingleScatteringDataGridded(frequencies,
                                           t_grid_,
                                           phase_matrix,
                                           extinction_matrix,
                                           absorption_vector,
                                           backscattering_coeff,
                                           forwardscattering_coeff);
  }

  SingleScatteringDataImpl *interpolate_temperature(
      eigen::VectorPtr<Scalar> temperatures) {
    auto phase_matrix = ScatteringDataFieldGridded<Scalar>(
        phase_matrix_.interpolate_temperature(temperatures));
    auto extinction_matrix = ScatteringDataFieldGridded<Scalar>(
        extinction_matrix_.interpolate_temperature(temperatures));
    auto absorption_vector = ScatteringDataFieldGridded<Scalar>(
        absorption_vector_.interpolate_temperature(temperatures));

    using Regridder = RegularRegridder<Scalar, 1>;
    Regridder scat_coeff_regridder({*t_grid_}, {*temperatures});
    auto backscattering_coeff = std::make_shared<ScatteringCoeff>(
        scat_coeff_regridder.regrid(*backscattering_coeff_));
    auto forwardscattering_coeff = std::make_shared<ScatteringCoeff>(
        scat_coeff_regridder.regrid(*backscattering_coeff_));
    return new SingleScatteringDataGridded(f_grid_,
                                           temperatures,
                                           phase_matrix,
                                           extinction_matrix,
                                           absorption_vector,
                                           backscattering_coeff,
                                           forwardscattering_coeff);
  }

  SingleScatteringDataImpl *interpolate_angles(
      eigen::VectorPtr<Scalar> lon_inc,
      eigen::VectorPtr<Scalar> lat_inc,
      eigen::VectorPtr<Scalar> lon_scat,
      eigen::VectorPtr<Scalar> lat_scat) {
    auto phase_matrix = ScatteringDataFieldGridded<Scalar>(
        phase_matrix_.interpolate_angles(lon_inc, lat_inc, lon_scat, lat_scat));
    auto extinction_matrix = ScatteringDataFieldGridded<Scalar>(
        extinction_matrix_.interpolate_angles(lon_inc,
                                              lat_inc,
                                              lon_scat,
                                              lat_scat));
    auto absorption_vector = ScatteringDataFieldGridded<Scalar>(
        absorption_vector_.interpolate_angles(lon_inc,
                                              lat_inc,
                                              lon_scat,
                                              lat_scat));

    return new SingleScatteringDataGridded(f_grid_,
                                           t_grid_,
                                           phase_matrix,
                                           extinction_matrix,
                                           absorption_vector,
                                           backscattering_coeff_,
                                           forwardscattering_coeff_);
  }

  void operator+=(const SingleScatteringDataImpl *other) {
    auto converted = other->to_gridded();
    this->SingleScatteringDataBase<Scalar>::operator+=(*converted);
    phase_matrix_ += converted->phase_matrix_;
    extinction_matrix_ += converted->extinction_matrix_;
    absorption_vector_ += converted->absorption_vector_;
  }

  SingleScatteringDataImpl *operator+(const SingleScatteringDataImpl *other) {
    auto result = new SingleScatteringDataGridded(this->copy());
    result->operator+=(other);
    return result;
  }

  detail::ConversionPtr<const SingleScatteringDataGridded> to_gridded() const {
    return detail::make_conversion_ptr(this, false);
  }

  // explicit operator SingeScatteringDataSpectral();
  // explicit operator SingeScatteringDataFullySpectral();

 private:
  VectorPtr dummy_grid_ = std::make_shared<Vector>(1);
  VectorPtr lon_inc_, lat_inc_, lon_scat_, lat_scat_;
  ScatteringDataFieldGridded<Scalar> phase_matrix_;
  ScatteringDataFieldGridded<Scalar> extinction_matrix_;
  ScatteringDataFieldGridded<Scalar> absorption_vector_;
};

// pxx :: hide
template <typename Scalar>
class SingleScatteringDataSpectral {};

// pxx :: hide
template <typename Scalar>
class SingleScatteringDataFullySpectral {};

////////////////////////////////////////////////////////////////////////////////
// Member function definitions.
////////////////////////////////////////////////////////////////////////////////

SingleScatteringData::SingleScatteringData(
    eigen::VectorPtr<double> f_grid,
    eigen::VectorPtr<double> t_grid,
    eigen::VectorPtr<double> lon_inc,
    eigen::VectorPtr<double> lat_inc,
    eigen::VectorPtr<double> lon_scat,
    eigen::VectorPtr<double> lat_scat,
    eigen::TensorPtr<double, 7> phase_matrix,
    eigen::TensorPtr<double, 7> extinction_matrix,
    eigen::TensorPtr<double, 7> absorption_vector,
    eigen::TensorPtr<double, 4> backscattering_coeff,
    eigen::TensorPtr<double, 4> forwardscattering_coeff)
    : data_(new SingleScatteringDataGridded<double>(f_grid,
                                                    t_grid,
                                                    lon_inc,
                                                    lat_inc,
                                                    lon_scat,
                                                    lat_scat,
                                                    phase_matrix,
                                                    extinction_matrix,
                                                    absorption_vector,
                                                    backscattering_coeff,
                                                    forwardscattering_coeff)) {}

}  // namespace scatlib

#endif
