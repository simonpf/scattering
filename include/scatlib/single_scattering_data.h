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
  Random = 0,
  AzimuthallyRandom = 1,
  General = 2
};

namespace detail {

Index get_n_phase_matrix_elements(ParticleType type) {
  switch (type) {
    case ParticleType::Random:
      return 6;
    case ParticleType::AzimuthallyRandom:
      return 16;
    case ParticleType::General:
      return 16;
  }
}

Index get_n_extinction_matrix_elements(ParticleType type) {
  switch (type) {
    case ParticleType::Random:
      return 1;
    case ParticleType::AzimuthallyRandom:
      return 3;
    case ParticleType::General:
      return 4;
  }
}

Index get_n_absorption_vector_elements(ParticleType type) {
    switch (type) {
    case ParticleType::Random:
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
  virtual eigen::Tensor<double, 2> get_forward_scattering_coeff() = 0;
  virtual eigen::Tensor<double, 2> get_backward_scattering_coeff() = 0;

  // Addition
  virtual void operator+=(const SingleScatteringDataImpl *other) = 0;
  virtual SingleScatteringDataImpl *operator+(
      const SingleScatteringDataImpl *other) = 0;

  // Scaling
  virtual void operator*=(double c) = 0;
  virtual SingleScatteringDataImpl *operator*(double c) = 0;

  virtual detail::ConversionPtr<const SingleScatteringDataGridded<double>> to_gridded() const = 0;
  virtual detail::ConversionPtr<const SingleScatteringDataSpectral<double>> to_spectral() const = 0;

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
   * @param backward_scattering_coeff Tensor containing the backward_scattering
   * coefficients
   * @param forward_scattering_coeff Tensor containing the forward_scattering
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
                         scatlib::eigen::TensorPtr<double, 4> backward_scattering_coeff,
                         scatlib::eigen::TensorPtr<double, 4> forward_scattering_coeff);

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
        scatlib::eigen::Tensor<double, 4> backward_scattering_coeff,
        scatlib::eigen::Tensor<double, 4> forward_scattering_coeff)
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
              std::make_shared<eigen::Tensor<double, 4>>(backward_scattering_coeff),
              std::make_shared<eigen::Tensor<double, 4>>(
                  forward_scattering_coeff)) {}

    SingleScatteringData(scatlib::eigen::Vector<double> f_grid,
                         scatlib::eigen::Vector<double> t_grid,
                         scatlib::eigen::Vector<double> lon_inc,
                         scatlib::eigen::Vector<double> lat_inc,
                         scatlib::eigen::Vector<double> lon_scat,
                         scatlib::eigen::Vector<double> lat_scat,
                         ParticleType type);


  /** Create from spectral scattering data.
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
   * @param backward_scattering_coeff Tensor containing the backward_scattering
   * coefficients
   * @param forward_scattering_coeff Tensor containing the forward_scattering
   * coefficients.
   */
    // pxx :: hide
    SingleScatteringData(scatlib::eigen::VectorPtr<double> f_grid,
                         scatlib::eigen::VectorPtr<double> t_grid,
                         scatlib::eigen::VectorPtr<double> lon_inc,
                         scatlib::eigen::VectorPtr<double> lat_inc,
                         std::shared_ptr<sht::SHT> sht_scat,
                         scatlib::eigen::TensorPtr<double, 6> phase_matrix,
                         scatlib::eigen::TensorPtr<double, 6> extinction_matrix,
                         scatlib::eigen::TensorPtr<double, 6> absorption_vector,
                         scatlib::eigen::TensorPtr<double, 4> backward_scattering_coeff,
                         scatlib::eigen::TensorPtr<double, 4> forward_scattering_coeff);

    SingleScatteringData(scatlib::eigen::Vector<double> f_grid,
                         scatlib::eigen::Vector<double> t_grid,
                         scatlib::eigen::Vector<double> lon_inc,
                         scatlib::eigen::Vector<double> lat_inc,
                         Index l_max,
                         ParticleType type);

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
  eigen::Tensor<double, 2> get_forward_scattering_coeff() {
      return data_->get_forward_scattering_coeff();
  }
  eigen::Tensor<double, 2> get_backward_scattering_coeff() {
      return data_->get_backward_scattering_coeff();
  }

  // Addition
  SingleScatteringData &operator+=(const SingleScatteringData &other) {
    data_->operator+=(other.data_);
    return *this;
  }
  SingleScatteringData operator+(const SingleScatteringData &other) {
    return SingleScatteringData(data_->operator+(other.data_));
  }

  // Scaling
  SingleScatteringData &operator*=(double c) {
      data_->operator*=(c);
      return *this;
  }
  SingleScatteringData operator*(double c) {
      return SingleScatteringData(data_->operator*(c));
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
                           eigen::TensorPtr<Scalar, 4> backward_scattering_coeff,
                           eigen::TensorPtr<Scalar, 4> forward_scattering_coeff)
      : n_freqs_(f_grid->size()),
        n_temps_(t_grid->size()),
        f_grid_(f_grid),
        t_grid_(t_grid),
        backward_scattering_coeff_(backward_scattering_coeff),
        forward_scattering_coeff_(forward_scattering_coeff) {}

  eigen::Tensor<Scalar, 2> get_backward_scattering_coeff() {
      auto index = std::array<Index, 2>{0, 0};
      return eigen::tensor_index(*backward_scattering_coeff_, index);
  }

  eigen::Tensor<Scalar, 2> get_forward_scattering_coeff() {
      auto index = std::array<Index, 2>{0, 0};
      return eigen::tensor_index(*forward_scattering_coeff_, index);
  }

  void set_data(Index f_index,
                Index t_index,
                const SingleScatteringDataBase &other) {
      auto dst_index = std::array<Index, 2>{f_index, t_index};
      auto src_index = std::array<Index, 2>{0, 0};

      eigen::tensor_index(*backward_scattering_coeff_, dst_index) =
          eigen::tensor_index(*other.backward_scattering_coeff_, src_index);
      eigen::tensor_index(*forward_scattering_coeff_, dst_index) =
          eigen::tensor_index(*other.forward_scattering_coeff_, src_index);
  }

  void operator+=(const SingleScatteringDataBase &other) {
    backward_scattering_coeff_->operator+=(*other.backward_scattering_coeff_);
    forward_scattering_coeff_->operator+=(*other.forward_scattering_coeff_);
  }

  void operator*=(Scalar c) {
      *backward_scattering_coeff_ = c * (*backward_scattering_coeff_);
      *forward_scattering_coeff_ = c * (*forward_scattering_coeff_);
  }

 protected:
  size_t n_freqs_, n_temps_;
  VectorPtr f_grid_;
  VectorPtr t_grid_;
  ScatteringCoeffPtr backward_scattering_coeff_;
  ScatteringCoeffPtr forward_scattering_coeff_;
};

template <typename Scalar>
class SingleScatteringDataGridded : public SingleScatteringDataBase<Scalar>,
                                    public SingleScatteringDataImpl {
  using SingleScatteringDataBase<Scalar>::backward_scattering_coeff_;
  using SingleScatteringDataBase<Scalar>::f_grid_;
  using SingleScatteringDataBase<Scalar>::forward_scattering_coeff_;
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
      ScatteringCoeffPtr backward_scattering_coeff,
      ScatteringCoeffPtr forward_scattering_coeff)

      : SingleScatteringDataBase<Scalar>(f_grid,
                                         t_grid,
                                         backward_scattering_coeff,
                                         forward_scattering_coeff),
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
                              ScatteringCoeffPtr backward_scattering_coeff,
                              ScatteringCoeffPtr forward_scattering_coeff)
      : SingleScatteringDataBase<Scalar>(f_grid,
                                         t_grid,
                                         backward_scattering_coeff,
                                         forward_scattering_coeff),
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
    SingleScatteringDataBase<Scalar>::set_data(f_index,
                                               t_index,
                                               *converted);
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

  eigen::Tensor<Scalar, 2> get_backward_scattering_coeff() {
      return SingleScatteringDataBase<Scalar>::get_backward_scattering_coeff();
  }

  eigen::Tensor<Scalar, 2> get_forward_scattering_coeff() {
      return SingleScatteringDataBase<Scalar>::get_forward_scattering_coeff();
  }

  SingleScatteringDataGridded copy() {
    return SingleScatteringDataGridded(
        std::make_shared<Vector>(*f_grid_),
        std::make_shared<Vector>(*t_grid_),
        phase_matrix_.copy(),
        extinction_matrix_.copy(),
        absorption_vector_.copy(),
        std::make_shared<ScatteringCoeff>(*backward_scattering_coeff_),
        std::make_shared<ScatteringCoeff>(*forward_scattering_coeff_));
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
    auto backward_scattering_coeff = std::make_shared<ScatteringCoeff>(
        scat_coeff_regridder.regrid(*backward_scattering_coeff_));
    auto forward_scattering_coeff = std::make_shared<ScatteringCoeff>(
        scat_coeff_regridder.regrid(*forward_scattering_coeff_));
    return new SingleScatteringDataGridded(frequencies,
                                           t_grid_,
                                           phase_matrix,
                                           extinction_matrix,
                                           absorption_vector,
                                           backward_scattering_coeff,
                                           forward_scattering_coeff);
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
    auto backward_scattering_coeff = std::make_shared<ScatteringCoeff>(
        scat_coeff_regridder.regrid(*backward_scattering_coeff_));
    auto forward_scattering_coeff = std::make_shared<ScatteringCoeff>(
        scat_coeff_regridder.regrid(*forward_scattering_coeff_));
    return new SingleScatteringDataGridded(f_grid_,
                                           temperatures,
                                           phase_matrix,
                                           extinction_matrix,
                                           absorption_vector,
                                           backward_scattering_coeff,
                                           forward_scattering_coeff);
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
                                           backward_scattering_coeff_,
                                           forward_scattering_coeff_);
  }

  void operator+=(const SingleScatteringDataImpl *other) {
    auto converted = other->to_gridded();
    this->SingleScatteringDataBase<Scalar>::operator+=(*converted);
    phase_matrix_ += converted->phase_matrix_;
    extinction_matrix_ += converted->extinction_matrix_;
    absorption_vector_ += converted->absorption_vector_;
  }

  SingleScatteringDataImpl * operator+(const SingleScatteringDataImpl *other) {
    auto result = new SingleScatteringDataGridded(this->copy());
    result->operator+=(other);
    return result;
  }

  void operator*=(Scalar c) {
      this->SingleScatteringDataBase<Scalar>::operator*=(c);
      phase_matrix_ *= c;
      extinction_matrix_ *= c;
      absorption_vector_ *= c;
  }

  SingleScatteringDataImpl * operator*(Scalar c) {
      auto result = new SingleScatteringDataGridded(this->copy());
      result->operator*=(c);
      return result;
  }

  detail::ConversionPtr<const SingleScatteringDataGridded> to_gridded() const {
    return detail::make_conversion_ptr(this, false);
  }

  detail::ConversionPtr<const SingleScatteringDataSpectral<Scalar>> to_spectral() const;

  // explicit operator SingeScatteringDataSpectral();
  // explicit operator SingeScatteringDataFullySpectral();

 private:
  VectorPtr dummy_grid_ = std::make_shared<Vector>(1);
  VectorPtr lon_inc_, lat_inc_, lon_scat_, lat_scat_;
  ScatteringDataFieldGridded<Scalar> phase_matrix_;
  ScatteringDataFieldGridded<Scalar> extinction_matrix_;
  ScatteringDataFieldGridded<Scalar> absorption_vector_;
};

////////////////////////////////////////////////////////////////////////////////
// Spectral single scattering data.
////////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
class SingleScatteringDataSpectral : public SingleScatteringDataBase<Scalar>,
    public SingleScatteringDataImpl {
  using SingleScatteringDataBase<Scalar>::backward_scattering_coeff_;
  using SingleScatteringDataBase<Scalar>::f_grid_;
  using SingleScatteringDataBase<Scalar>::forward_scattering_coeff_;
  using SingleScatteringDataBase<Scalar>::n_freqs_;
  using SingleScatteringDataBase<Scalar>::n_temps_;
  using SingleScatteringDataBase<Scalar>::t_grid_;

 public:
  using Vector = eigen::Vector<Scalar>;
  using VectorPtr = std::shared_ptr<Vector>;
  using ShtPtr = std::shared_ptr<sht::SHT>;
  using DataTensor = eigen::Tensor<Scalar, 6>;
  using DataPtr = std::shared_ptr<DataTensor>;
  using ScatteringCoeff = eigen::Tensor<Scalar, 4>;
  using ScatteringCoeffPtr = std::shared_ptr<ScatteringCoeff>;
  using OtherScalar =
      std::conditional<std::is_same<Scalar, double>::value, float, double>;

  SingleScatteringDataSpectral(
      VectorPtr f_grid,
      VectorPtr t_grid,
      ScatteringDataFieldSpectral<Scalar> phase_matrix,
      ScatteringDataFieldSpectral<Scalar> extinction_matrix,
      ScatteringDataFieldSpectral<Scalar> absorption_vector,
      ScatteringCoeffPtr backward_scattering_coeff,
      ScatteringCoeffPtr forward_scattering_coeff)

      : SingleScatteringDataBase<Scalar>(f_grid,
                                         t_grid,
                                         backward_scattering_coeff,
                                         forward_scattering_coeff),
        phase_matrix_(phase_matrix),
        extinction_matrix_(extinction_matrix),
        absorption_vector_(absorption_vector) {}

  SingleScatteringDataSpectral(VectorPtr f_grid,
                               VectorPtr t_grid,
                               VectorPtr lon_inc,
                               VectorPtr lat_inc,
                               ShtPtr sht_scat,
                               DataPtr phase_matrix,
                               DataPtr extinction_matrix,
                               DataPtr absorption_vector,
                               ScatteringCoeffPtr backward_scattering_coeff,
                               ScatteringCoeffPtr forward_scattering_coeff)
      : SingleScatteringDataBase<Scalar>(f_grid,
                                         t_grid,
                                         backward_scattering_coeff,
                                         forward_scattering_coeff),
        phase_matrix_{f_grid,
                      t_grid,
                      lon_inc,
                      lat_inc,
          sht_scat,

                      phase_matrix},
        extinction_matrix_{f_grid,
                           t_grid,
                           lon_inc,
                           lat_inc,
                sht_scat,
                           extinction_matrix},
        absorption_vector_{f_grid,
                           t_grid,
                           lon_inc,
                           lat_inc,
                sht_scat,
                           absorption_vector} {}

  eigen::Vector<double> get_f_grid() { return *f_grid_; }
  eigen::Vector<double> get_t_grid() { return *t_grid_; }

  void set_data(Index f_index,
                Index t_index,
                const SingleScatteringDataImpl &other) {
    auto converted = other.to_spectral();
    SingleScatteringDataBase<Scalar>::set_data(f_index,
                                               t_index,
                                               *converted);
    phase_matrix_.set_data(f_index, t_index, converted->phase_matrix_);
    extinction_matrix_.set_data(f_index,
                                t_index,
                                converted->extinction_matrix_);
    absorption_vector_.set_data(f_index,
                                t_index,
                                converted->absorption_vector_);
  }

  eigen::Tensor<Scalar, 5> get_phase_matrix() {
    auto phase_matrix_gridded_ = phase_matrix_.to_spectral();
    return eigen::tensor_index<2>(phase_matrix_gridded_.get_data(), {0, 0});
  }

  eigen::Tensor<Scalar, 5> get_extinction_matrix() {
    return eigen::tensor_index<2>(extinction_matrix_.get_data(), {0, 0});
  }

  eigen::Tensor<Scalar, 5> get_absorption_vector() {
    return eigen::tensor_index<2>(absorption_vector_.get_data(), {0, 0});
  }

  eigen::Tensor<Scalar, 2> get_backward_scattering_coeff() {
      return SingleScatteringDataBase<Scalar>::get_backward_scattering_coeff();
  }

  eigen::Tensor<Scalar, 2> get_forward_scattering_coeff() {
      return SingleScatteringDataBase<Scalar>::get_forward_scattering_coeff();
  }

  SingleScatteringDataSpectral copy() {
    return SingleScatteringDataSpectral(
        std::make_shared<Vector>(*f_grid_),
        std::make_shared<Vector>(*t_grid_),
        phase_matrix_.copy(),
        extinction_matrix_.copy(),
        absorption_vector_.copy(),
        std::make_shared<ScatteringCoeff>(*backward_scattering_coeff_),
        std::make_shared<ScatteringCoeff>(*forward_scattering_coeff_));
  }

  SingleScatteringDataImpl *interpolate_frequency(
      eigen::VectorPtr<double> frequencies) {
    auto phase_matrix = ScatteringDataFieldSpectral<Scalar>(
        phase_matrix_.interpolate_frequency(frequencies));
    auto extinction_matrix = ScatteringDataFieldSpectral<Scalar>(
        extinction_matrix_.interpolate_frequency(frequencies));
    auto absorption_vector = ScatteringDataFieldSpectral<Scalar>(
        absorption_vector_.interpolate_frequency(frequencies));

    using Regridder = RegularRegridder<Scalar, 0>;
    Regridder scat_coeff_regridder({*f_grid_}, {*frequencies});
    auto backward_scattering_coeff = std::make_shared<ScatteringCoeff>(
        scat_coeff_regridder.regrid(*backward_scattering_coeff_));
    auto forward_scattering_coeff = std::make_shared<ScatteringCoeff>(
        scat_coeff_regridder.regrid(*forward_scattering_coeff_));
    return new SingleScatteringDataSpectral(frequencies,
                                           t_grid_,
                                           phase_matrix,
                                           extinction_matrix,
                                           absorption_vector,
                                           backward_scattering_coeff,
                                           forward_scattering_coeff);
  }

  SingleScatteringDataImpl *interpolate_temperature(
      eigen::VectorPtr<Scalar> temperatures) {
    auto phase_matrix = ScatteringDataFieldSpectral<Scalar>(
        phase_matrix_.interpolate_temperature(temperatures));
    auto extinction_matrix = ScatteringDataFieldSpectral<Scalar>(
        extinction_matrix_.interpolate_temperature(temperatures));
    auto absorption_vector = ScatteringDataFieldSpectral<Scalar>(
        absorption_vector_.interpolate_temperature(temperatures));

    using Regridder = RegularRegridder<Scalar, 1>;
    Regridder scat_coeff_regridder({*t_grid_}, {*temperatures});
    auto backward_scattering_coeff = std::make_shared<ScatteringCoeff>(
        scat_coeff_regridder.regrid(*backward_scattering_coeff_));
    auto forward_scattering_coeff = std::make_shared<ScatteringCoeff>(
        scat_coeff_regridder.regrid(*forward_scattering_coeff_));
    return new SingleScatteringDataSpectral(f_grid_,
                                           temperatures,
                                           phase_matrix,
                                           extinction_matrix,
                                           absorption_vector,
                                           backward_scattering_coeff,
                                           forward_scattering_coeff);
  }

  SingleScatteringDataImpl *interpolate_angles(
      eigen::VectorPtr<Scalar> lon_inc,
      eigen::VectorPtr<Scalar> lat_inc,
      eigen::VectorPtr<Scalar> /*lon_scat*/,
      eigen::VectorPtr<Scalar> /*lat_scat*/) {
    auto phase_matrix = ScatteringDataFieldSpectral<Scalar>(
        phase_matrix_.interpolate_angles(lon_inc, lat_inc));
    auto extinction_matrix = ScatteringDataFieldSpectral<Scalar>(
        extinction_matrix_.interpolate_angles(lon_inc, lat_inc));
    auto absorption_vector = ScatteringDataFieldSpectral<Scalar>(
        absorption_vector_.interpolate_angles(lon_inc, lat_inc));

    return new SingleScatteringDataSpectral(f_grid_,
                                           t_grid_,
                                           phase_matrix,
                                           extinction_matrix,
                                           absorption_vector,
                                           backward_scattering_coeff_,
                                           forward_scattering_coeff_);
  }

  void operator+=(const SingleScatteringDataImpl *other) {
    auto converted = other->to_spectral();
    this->SingleScatteringDataBase<Scalar>::operator+=(*converted);
    phase_matrix_ += converted->phase_matrix_;
    extinction_matrix_ += converted->extinction_matrix_;
    absorption_vector_ += converted->absorption_vector_;
  }

  SingleScatteringDataImpl * operator+(const SingleScatteringDataImpl *other) {
    auto result = new SingleScatteringDataSpectral(this->copy());
    result->operator+=(other);
    return result;
  }

  void operator*=(Scalar c) {
      this->SingleScatteringDataBase<Scalar>::operator*=(c);
      phase_matrix_ *= c;
      extinction_matrix_ *= c;
      absorption_vector_ *= c;
  }

  SingleScatteringDataImpl * operator*(Scalar c) {
      auto result = new SingleScatteringDataSpectral(this->copy());
      result->operator*=(c);
      return result;
  }

  detail::ConversionPtr<const SingleScatteringDataGridded<Scalar>> to_gridded() const;

  detail::ConversionPtr<const SingleScatteringDataSpectral> to_spectral() const {
    return detail::make_conversion_ptr(this, false);
  }

  // explicit operator SingeScatteringDataSpectral();
  // explicit operator SingeScatteringDataFullySpectral();

 private:
  VectorPtr lon_inc_, lat_inc_, lon_scat_, lat_scat_;
  ScatteringDataFieldSpectral<Scalar> phase_matrix_;
  ScatteringDataFieldSpectral<Scalar> extinction_matrix_;
  ScatteringDataFieldSpectral<Scalar> absorption_vector_;
};

// pxx :: hide
template <typename Scalar>
class SingleScatteringDataFullySpectral {};

////////////////////////////////////////////////////////////////////////////////
// Member function definitions.
////////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
    detail::ConversionPtr<const SingleScatteringDataSpectral<Scalar>>
    SingleScatteringDataGridded<Scalar>::to_spectral() const {
    auto phase_matrix =
        ScatteringDataFieldSpectral<Scalar>(phase_matrix_.to_spectral());
    return new SingleScatteringDataSpectral<Scalar>(f_grid_,
                                                    t_grid_,
                                                    phase_matrix,
                                                    extinction_matrix_,
                                                    absorption_vector_,
                                                    backward_scattering_coeff_,
                                                    forward_scattering_coeff_);
}

template <typename Scalar>
detail::ConversionPtr<const SingleScatteringDataGridded<Scalar>>
SingleScatteringDataSpectral<Scalar>::to_gridded() const {
  auto phase_matrix =
      ScatteringDataFieldSpectral<Scalar>(phase_matrix_.to_gridded());
  return new SingleScatteringDataGridded<Scalar>(f_grid_,
                                                 t_grid_,
                                                 phase_matrix,
                                                 extinction_matrix_,
                                                 absorption_vector_,
                                                 backward_scattering_coeff_,
                                                 forward_scattering_coeff_);
}

//
// Gridded
//

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
    eigen::TensorPtr<double, 4> backward_scattering_coeff,
    eigen::TensorPtr<double, 4> forward_scattering_coeff)
    : data_(new SingleScatteringDataGridded<double>(f_grid,
                                                    t_grid,
                                                    lon_inc,
                                                    lat_inc,
                                                    lon_scat,
                                                    lat_scat,
                                                    phase_matrix,
                                                    extinction_matrix,
                                                    absorption_vector,
                                                    backward_scattering_coeff,
                                                    forward_scattering_coeff)) {}

SingleScatteringData::SingleScatteringData(scatlib::eigen::Vector<double> f_grid,
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

//
// Spectral
//

SingleScatteringData::SingleScatteringData(
    eigen::VectorPtr<double> f_grid,
    eigen::VectorPtr<double> t_grid,
    eigen::VectorPtr<double> lon_inc,
    eigen::VectorPtr<double> lat_inc,
    std::shared_ptr<sht::SHT> sht_scat,
    eigen::TensorPtr<double, 6> phase_matrix,
    eigen::TensorPtr<double, 6> extinction_matrix,
    eigen::TensorPtr<double, 6> absorption_vector,
    eigen::TensorPtr<double, 4> backward_scattering_coeff,
    eigen::TensorPtr<double, 4> forward_scattering_coeff)
    : data_(new SingleScatteringDataSpectral<double>(f_grid,
                                                     t_grid,
                                                     lon_inc,
                                                     lat_inc,
                                                     sht_scat,
                                                     phase_matrix,
                                                     extinction_matrix,
                                                     absorption_vector,
                                                     backward_scattering_coeff,
                                                     forward_scattering_coeff)) {}

SingleScattering::SingleScatteringData(scatlib::eigen::Vector<double> f_grid,
                                       scatlib::eigen::Vector<double> t_grid,
                                       scatlib::eigen::Vector<double> lon_inc,
                                       scatlib::eigen::Vector<double> lat_inc,
                                       Index l_max,
                                       ParticleType type)
    : SingleScatteringData(
          std::make_shared<eigen::Vector<double>>(f_grid),
          std::make_shared<eigen::Vector<double>>(t_grid),
          std::make_shared<eigen::Vector<double>>(lon_inc),
          std::make_shared<eigen::Vector<double>>(lat_inc),
          std::make_shared<sht::SHT>(l_max, l_max),
          std::make_shared<eigen::Tensor<double, 6>>(std::array<Index, 6>{
              f_grid.size(),
              t_grid.size(),
              lon_inc.size(),
              lat_inc.size(),
              sht::SHT::calc_n_spectral_coeffs(l_max, l_max);
              detail::get_n_phase_matrix_elements(type)}),
          std::make_shared<eigen::Tensor<double, 6>>(std::array<Index, 6>{
              f_grid.size(),
              t_grid.size(),
              lon_inc.size(),
              lat_inc.size(),
              1,
              detail::get_n_extinction_matrix_elements(type)}),
          std::make_shared<eigen::Tensor<double, 6>>(std::array<Index, 6>{
              f_grid.size(),
              t_grid.size(),
              lon_inc.size(),
              lat_inc.size(),
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

}  // namespace scatlib

#endif
