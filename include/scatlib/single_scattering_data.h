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
#include <scatlib/interpolation.h>
#include <scatlib/scattering_data_field.h>
#include <scatlib/sht.h>

#include <cassert>
#include <memory>

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
enum class ParticleType { Random = 0, AzimuthallyRandom = 1, General = 2 };

namespace detail {

inline Index get_n_phase_matrix_elements(ParticleType type) {
  switch (type) {
    case ParticleType::Random:
      return 6;
    case ParticleType::AzimuthallyRandom:
      return 16;
    case ParticleType::General:
      return 16;
  }
  return 1;
}

inline Index get_n_extinction_matrix_elements(ParticleType type) {
  switch (type) {
    case ParticleType::Random:
      return 1;
    case ParticleType::AzimuthallyRandom:
      return 3;
    case ParticleType::General:
      return 4;
  }
  return 1;
}

inline Index get_n_absorption_vector_elements(ParticleType type) {
  switch (type) {
    case ParticleType::Random:
      return 1;
    case ParticleType::AzimuthallyRandom:
      return 2;
    case ParticleType::General:
      return 4;
  }
  return 1;
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
  virtual ~SingleScatteringDataImpl() = default;

  virtual void set_data(Index f_index,
                        Index t_index,
                        const SingleScatteringDataImpl &) = 0;

  virtual SingleScatteringDataImpl *copy() = 0;

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
  virtual eigen::Vector<double> get_lon_inc() = 0;
  virtual eigen::Vector<double> get_lat_inc() = 0;
  virtual eigen::Vector<double> get_lon_scat() = 0;
  virtual eigen::Vector<double> get_lat_scat() = 0;

  virtual eigen::Index get_n_lon_inc() = 0;
  virtual eigen::Index get_n_lat_inc() = 0;
  virtual eigen::Index get_n_lon_scat() = 0;
  virtual eigen::Index get_n_lat_scat() = 0;

  // Data access.
  virtual eigen::Tensor<double, 7> get_phase_matrix() const = 0;
  virtual eigen::Tensor<std::complex<double>, 6>
  get_phase_matrix_spectral() const = 0;
  virtual eigen::Tensor<double, 7> get_extinction_matrix() const = 0;
  virtual eigen::Tensor<double, 7> get_absorption_vector() const = 0;
  virtual eigen::Tensor<double, 7> get_forward_scattering_coeff() const = 0;
  virtual eigen::Tensor<double, 7> get_backward_scattering_coeff() const = 0;

  // Addition
  virtual void operator+=(const SingleScatteringDataImpl *other) = 0;
  virtual SingleScatteringDataImpl *operator+(
      const SingleScatteringDataImpl *other) = 0;

  // Scaling
  virtual void operator*=(double c) = 0;
  virtual SingleScatteringDataImpl *operator*(double c) = 0;
  virtual void normalize(double norm) = 0;

  // Regridding
  virtual SingleScatteringDataImpl *regrid() = 0;

  virtual void set_number_of_scattering_coeffs(Index n) = 0;

  virtual detail::ConversionPtr<const SingleScatteringDataGridded<double>>
  to_gridded() const = 0;

  virtual detail::ConversionPtr<const SingleScatteringDataSpectral<double>>
  to_spectral(Index l_max, Index m_max) const = 0;
  virtual detail::ConversionPtr<const SingleScatteringDataSpectral<double>>
  to_spectral(Index l_max, Index m_max, Index n_lon, Index n_lat) const = 0;
  virtual detail::ConversionPtr<const SingleScatteringDataSpectral<double>>
  to_spectral() const = 0;

  // Conversion operators
  // virtual operator SingleScatteringDataGridded<float>() = 0;
  // virtual operator SingleScatteringDataGridded<double>() = 0;
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
  SingleScatteringData(SingleScatteringDataImpl *data) : data_(data) {}

  SingleScatteringData() {}

  SingleScatteringData(const SingleScatteringData &other) = default;

  SingleScatteringData copy() const { return SingleScatteringData(data_->copy()); }

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
  SingleScatteringData(
      scatlib::eigen::VectorPtr<double> f_grid,
      scatlib::eigen::VectorPtr<double> t_grid,
      scatlib::eigen::VectorPtr<double> lon_inc,
      scatlib::eigen::VectorPtr<double> lat_inc,
      scatlib::eigen::VectorPtr<double> lon_scat,
      scatlib::eigen::VectorPtr<double> lat_scat,
      scatlib::eigen::TensorPtr<double, 7> phase_matrix,
      scatlib::eigen::TensorPtr<double, 7> extinction_matrix,
      scatlib::eigen::TensorPtr<double, 7> absorption_vector,
      scatlib::eigen::TensorPtr<double, 7> backward_scattering_coeff,
      scatlib::eigen::TensorPtr<double, 7> forward_scattering_coeff);

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
      scatlib::eigen::Tensor<double, 7> backward_scattering_coeff,
      scatlib::eigen::Tensor<double, 7> forward_scattering_coeff)
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
            std::make_shared<eigen::Tensor<double, 7>>(
                backward_scattering_coeff),
            std::make_shared<eigen::Tensor<double, 7>>(
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
  SingleScatteringData(
      scatlib::eigen::VectorPtr<double> f_grid,
      scatlib::eigen::VectorPtr<double> t_grid,
      scatlib::eigen::VectorPtr<double> lon_inc,
      scatlib::eigen::VectorPtr<double> lat_inc,
      std::shared_ptr<sht::SHT> sht_scat,
      scatlib::eigen::TensorPtr<std::complex<double>, 6> phase_matrix,
      scatlib::eigen::TensorPtr<std::complex<double>, 6> extinction_matrix,
      scatlib::eigen::TensorPtr<std::complex<double>, 6> absorption_vector,
      scatlib::eigen::TensorPtr<std::complex<double>, 6>
          backward_scattering_coeff,
      scatlib::eigen::TensorPtr<std::complex<double>, 6>
          forward_scattering_coeff);

  SingleScatteringData(
      scatlib::eigen::Vector<double> f_grid,
      scatlib::eigen::Vector<double> t_grid,
      scatlib::eigen::Vector<double> lon_inc,
      scatlib::eigen::Vector<double> lat_inc,
      scatlib::sht::SHT sht_scat,
      scatlib::eigen::Tensor<std::complex<double>, 6> phase_matrix,
      scatlib::eigen::Tensor<std::complex<double>, 6> extinction_matrix,
      scatlib::eigen::Tensor<std::complex<double>, 6> absorption_vector,
      scatlib::eigen::Tensor<std::complex<double>, 6> backward_scattering_coeff,
      scatlib::eigen::Tensor<std::complex<double>, 6> forward_scattering_coeff);

  SingleScatteringData(scatlib::eigen::Vector<double> f_grid,
                       scatlib::eigen::Vector<double> t_grid,
                       scatlib::eigen::Vector<double> lon_inc,
                       scatlib::eigen::Vector<double> lat_inc,
                       Index l_max,
                       ParticleType type);

  eigen::Vector<double> get_f_grid() { return data_->get_f_grid(); }
  eigen::Vector<double> get_t_grid() { return data_->get_t_grid(); }
  eigen::Vector<double> get_lon_inc() { return data_->get_lon_inc(); }
  eigen::Vector<double> get_lat_inc() { return data_->get_lat_inc(); }
  eigen::Vector<double> get_lon_scat() { return data_->get_lon_scat(); }
  eigen::Vector<double> get_lat_scat() { return data_->get_lat_scat(); }

  eigen::Index get_n_lon_inc() { return data_->get_n_lon_inc(); }
  eigen::Index get_n_lat_inc() { return data_->get_n_lon_inc(); }
  eigen::Index get_n_lon_scat() { return data_->get_n_lon_inc(); }
  eigen::Index get_n_lat_scat() { return data_->get_n_lon_inc(); }

  void set_data(Index f_index,
                Index t_index,
                const SingleScatteringData &other) {
    data_->set_data(f_index, t_index, *other.data_);
  }

  // Interpolation functions

  SingleScatteringData interpolate_frequency(
      eigen::Vector<double> frequencies) {
    auto result = data_->interpolate_frequency(
        std::make_shared<eigen::Vector<double>>(frequencies));
    return SingleScatteringData(std::move(result));
  }
  // pxx :: hide
  SingleScatteringData interpolate_frequency(
      std::shared_ptr<eigen::Vector<double>> frequencies) {
      auto result = data_->interpolate_frequency(frequencies);
      return SingleScatteringData(std::move(result));
  }

  SingleScatteringData interpolate_temperature(
      eigen::Vector<double> temperatures) {
    auto result = data_->interpolate_temperature(
        std::make_shared<eigen::Vector<double>>(temperatures));
    return SingleScatteringData(std::move(result));
  }
  // pxx :: hide
  SingleScatteringData interpolate_temperature(
      std::shared_ptr<eigen::Vector<double>> temperatures) {
      auto result = data_->interpolate_temperature(temperatures);
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

  // pxx :: hide
  SingleScatteringData interpolate_angles(std::shared_ptr<eigen::Vector<double>> lon_inc,
                                          std::shared_ptr<eigen::Vector<double>> lat_inc,
                                          std::shared_ptr<eigen::Vector<double>> lon_scat,
                                          std::shared_ptr<eigen::Vector<double>> lat_scat) {
      auto result = data_->interpolate_angles(lon_inc,
                                              lat_inc,
                                              lon_scat,
                                              lat_scat);
      return SingleScatteringData(std::move(result));
  }

  // Data access.
  eigen::Tensor<double, 7> get_phase_matrix() const {
    return data_->get_phase_matrix();
  }

  eigen::Tensor<std::complex<double>, 6> get_phase_matrix_spectral() const {
    return data_->get_phase_matrix_spectral();
  }

  eigen::Tensor<double, 7> get_extinction_matrix() const {
    return data_->get_extinction_matrix();
  }
  eigen::Tensor<double, 7> get_absorption_vector() const {
    return data_->get_absorption_vector();
  }
  eigen::Tensor<double, 7> get_forward_scattering_coeff() const {
    return data_->get_forward_scattering_coeff();
  }
  eigen::Tensor<double, 7> get_backward_scattering_coeff() const {
    return data_->get_backward_scattering_coeff();
  }

  // Addition
  SingleScatteringData &operator+=(const SingleScatteringData &other) {
    data_->operator+=(other.data_.get());
    return *this;
  }
  SingleScatteringData operator+(const SingleScatteringData &other) {
    return SingleScatteringData(data_->operator+(other.data_.get()));
  }

  // Scaling
  SingleScatteringData &operator*=(double c) {
    data_->operator*=(c);
    return *this;
  }
  SingleScatteringData operator*(double c) {
    return SingleScatteringData(data_->operator*(c));
  }

  void normalize(double norm) { data_->normalize(norm); }

  // Regrid
  SingleScatteringData regrid() { return data_->regrid(); }

  void set_number_of_scattering_coeffs(Index n) {
      data_->set_number_of_scattering_coeffs(n);
  }

  // Conversion
  inline SingleScatteringData to_gridded() const;
  inline SingleScatteringData to_spectral() const;
  inline SingleScatteringData to_spectral(Index l_max, Index m_max) const;
  inline SingleScatteringData to_spectral(Index l_max,
                                          Index m_max,
                                          Index n_lon,
                                          Index n_lat) const;

 private:
  std::shared_ptr<SingleScatteringDataImpl> data_;
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
                           eigen::VectorPtr<Scalar> t_grid)
      : n_freqs_(f_grid->size()),
        n_temps_(t_grid->size()),
        f_grid_(f_grid),
        t_grid_(t_grid) {}

 protected:
  size_t n_freqs_, n_temps_;
  VectorPtr f_grid_;
  VectorPtr t_grid_;
};

template <typename Scalar>
class SingleScatteringDataGridded : public SingleScatteringDataBase<Scalar>,
                                    public SingleScatteringDataImpl {
  using SingleScatteringDataBase<Scalar>::f_grid_;
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
      ScatteringDataFieldGridded<Scalar> backward_scattering_coeff,
      ScatteringDataFieldGridded<Scalar> forward_scattering_coeff)

      : SingleScatteringDataBase<Scalar>(f_grid, t_grid),
        phase_matrix_(phase_matrix),
        extinction_matrix_(extinction_matrix),
        absorption_vector_(absorption_vector),
        backward_scattering_coeff_(backward_scattering_coeff),
        forward_scattering_coeff_(forward_scattering_coeff) {}

  SingleScatteringDataGridded(VectorPtr f_grid,
                              VectorPtr t_grid,
                              VectorPtr lon_inc,
                              VectorPtr lat_inc,
                              VectorPtr lon_scat,
                              VectorPtr lat_scat,
                              DataPtr phase_matrix,
                              DataPtr extinction_matrix,
                              DataPtr absorption_vector,
                              DataPtr backward_scattering_coeff,
                              DataPtr forward_scattering_coeff)
      : SingleScatteringDataBase<Scalar>(f_grid, t_grid),
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
                           absorption_vector},
        backward_scattering_coeff_{f_grid,
                                   t_grid,
                                   lon_inc,
                                   lat_inc,
                                   dummy_grid_,
                                   dummy_grid_,
                                   backward_scattering_coeff},
        forward_scattering_coeff_(f_grid,
                                  t_grid,
                                  lon_inc,
                                  lat_inc,
                                  dummy_grid_,
                                  dummy_grid_,
                                  forward_scattering_coeff) {}

  eigen::Vector<double> get_f_grid() { return *f_grid_; }
  eigen::Vector<double> get_t_grid() { return *t_grid_; }
  eigen::Vector<double> get_lon_inc() { return phase_matrix_.get_lon_inc(); }
  eigen::Vector<double> get_lat_inc() { return phase_matrix_.get_lat_inc(); }
  eigen::Vector<double> get_lon_scat() { return phase_matrix_.get_lon_scat(); }
  eigen::Vector<double> get_lat_scat() { return phase_matrix_.get_lat_scat(); }

  eigen::Index get_n_lon_inc() { return phase_matrix_.get_n_lon_inc(); }
  eigen::Index get_n_lat_inc() { return phase_matrix_.get_n_lon_inc(); }
  eigen::Index get_n_lon_scat() { return phase_matrix_.get_n_lon_inc(); }
  eigen::Index get_n_lat_scat() { return phase_matrix_.get_n_lon_inc(); }

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
    forward_scattering_coeff_.set_data(f_index,
                                       t_index,
                                       converted->forward_scattering_coeff_);
    backward_scattering_coeff_.set_data(f_index,
                                        t_index,
                                        converted->backward_scattering_coeff_);
  }

  eigen::Tensor<Scalar, 7> get_phase_matrix() const {
    return phase_matrix_.get_data();
  }

  eigen::Tensor<std::complex<Scalar>, 6> get_phase_matrix_spectral() const {
    return phase_matrix_.to_spectral().get_data();
  }

  eigen::Tensor<Scalar, 7> get_extinction_matrix() const {
    return extinction_matrix_.get_data();
  }

  eigen::Tensor<Scalar, 7> get_absorption_vector() const {
    return absorption_vector_.get_data();
  }

  eigen::Tensor<Scalar, 7> get_backward_scattering_coeff() const {
    return backward_scattering_coeff_.get_data();
  }

  eigen::Tensor<Scalar, 7> get_forward_scattering_coeff() const {
      return forward_scattering_coeff_.get_data();
  }

  SingleScatteringDataGridded *copy() {
    return new SingleScatteringDataGridded(f_grid_,
                                           t_grid_,
                                           phase_matrix_.copy(),
                                           extinction_matrix_.copy(),
                                           absorption_vector_.copy(),
                                           backward_scattering_coeff_.copy(),
                                           forward_scattering_coeff_.copy());
  }

  SingleScatteringDataImpl *interpolate_frequency(
      eigen::VectorPtr<double> frequencies) {
    auto phase_matrix = ScatteringDataFieldGridded<Scalar>(
        phase_matrix_.interpolate_frequency(frequencies));
    auto extinction_matrix = ScatteringDataFieldGridded<Scalar>(
        extinction_matrix_.interpolate_frequency(frequencies));
    auto absorption_vector = ScatteringDataFieldGridded<Scalar>(
        absorption_vector_.interpolate_frequency(frequencies));
    auto backward_scattering_coeff = ScatteringDataFieldGridded<Scalar>(
        backward_scattering_coeff_.interpolate_frequency(frequencies));
    auto forward_scattering_coeff = ScatteringDataFieldGridded<Scalar>(
        forward_scattering_coeff_.interpolate_frequency(frequencies));

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
    auto backward_scattering_coeff = ScatteringDataFieldGridded<Scalar>(
        backward_scattering_coeff_.interpolate_temperature(temperatures));
    auto forward_scattering_coeff = ScatteringDataFieldGridded<Scalar>(
        forward_scattering_coeff_.interpolate_temperature(temperatures));

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
                                              dummy_grid_,
                                              dummy_grid_));
    auto absorption_vector = ScatteringDataFieldGridded<Scalar>(
        absorption_vector_.interpolate_angles(lon_inc,
                                              lat_inc,
                                              dummy_grid_,
                                              dummy_grid_));
    auto backward_scattering_coeff = ScatteringDataFieldGridded<Scalar>(
        backward_scattering_coeff_.interpolate_angles(lon_inc,
                                                      lat_inc,
                                                      dummy_grid_,
                                                      dummy_grid_));
    auto forward_scattering_coeff = ScatteringDataFieldGridded<Scalar>(
        forward_scattering_coeff_.interpolate_angles(lon_inc,
                                                     lat_inc,
                                                     dummy_grid_,
                                                     dummy_grid_));

    return new SingleScatteringDataGridded(f_grid_,
                                           t_grid_,
                                           phase_matrix,
                                           extinction_matrix,
                                           absorption_vector,
                                           backward_scattering_coeff,
                                           forward_scattering_coeff);
  }

  void operator+=(const SingleScatteringDataImpl *other) {
    auto converted = other->to_gridded();
    phase_matrix_ += converted->phase_matrix_;
    extinction_matrix_ += converted->extinction_matrix_;
    absorption_vector_ += converted->absorption_vector_;
    backward_scattering_coeff_ += converted->backward_scattering_coeff_;
    forward_scattering_coeff_ += converted->forward_scattering_coeff_;
  }

  SingleScatteringDataImpl *operator+(const SingleScatteringDataImpl *other) {
    auto result = this->copy();
    result->operator+=(other);
    return result;
  }

  void operator*=(Scalar c) {
    phase_matrix_ *= c;
    extinction_matrix_ *= c;
    absorption_vector_ *= c;
    backward_scattering_coeff_ *= c;
    forward_scattering_coeff_ *= c;
  }

  SingleScatteringDataImpl *operator*(Scalar c) {
    auto result = this->copy();
    result->operator*=(c);
    return result;
  }

  void normalize(Scalar norm) {
      phase_matrix_.normalize(norm);
      std::cout << "phase mat integrals: " << std::endl;
      std::cout << phase_matrix_.integrate_scattering_angles() << std::endl;
  }

  void set_number_of_scattering_coeffs(Index n) {
    phase_matrix_.set_number_of_scattering_coeffs(n);
    extinction_matrix_.set_number_of_scattering_coeffs(n);
    absorption_vector_.set_number_of_scattering_coeffs(n);
    backward_scattering_coeff_.set_number_of_scattering_coeffs(n);
    forward_scattering_coeff_.set_number_of_scattering_coeffs(n);
  }

  detail::ConversionPtr<const SingleScatteringDataGridded> to_gridded() const {
    return detail::make_conversion_ptr(this, false);
  }

  detail::ConversionPtr<const SingleScatteringDataSpectral<Scalar>> to_spectral(
      std::shared_ptr<sht::SHT>) const;
  detail::ConversionPtr<const SingleScatteringDataSpectral<Scalar>>
  to_spectral() const;
  detail::ConversionPtr<const SingleScatteringDataSpectral<Scalar>> to_spectral(
      Index l_max,
      Index m_max) const;
  detail::ConversionPtr<const SingleScatteringDataSpectral<Scalar>> to_spectral(
      Index l_max,
      Index m_max,
      Index n_lon,
      Index n_lat) const;

  // explicit operator SingeScatteringDataSpectral();
  // explicit operator SingeScatteringDataFullySpectral();

  SingleScatteringDataImpl * regrid() {
      auto n_lon_inc = phase_matrix_.get_n_lon_inc();
      auto n_lat_inc = phase_matrix_.get_n_lat_inc();
      auto n_lon_scat = phase_matrix_.get_n_lon_scat();
      auto n_lat_scat = phase_matrix_.get_n_lat_scat();
      n_lon_inc = std::max<Index>(n_lon_inc - n_lon_inc % 2, 1);
      n_lat_inc = std::max<Index>(n_lat_inc - n_lat_inc % 2, 1);
      n_lon_scat = std::max<Index>(n_lon_scat - n_lon_scat % 2, 1);
      n_lat_scat = std::max<Index>(n_lat_scat - n_lat_scat % 2, 1);

      auto lon_inc = std::make_shared<Vector>(sht::SHT::get_longitude_grid(n_lon_inc));
      auto lat_inc = std::make_shared<Vector>(sht::SHT::get_latitude_grid(n_lat_inc));
      auto lon_scat = std::make_shared<Vector>(sht::SHT::get_longitude_grid(n_lon_scat));
      auto lat_scat = std::make_shared<Vector>(sht::SHT::get_latitude_grid(n_lat_scat));

      auto phase_matrix = phase_matrix_.interpolate_angles(lon_inc, lat_inc, lon_scat, lat_scat);
      auto extinction_matrix = extinction_matrix_.interpolate_angles(lon_inc, lat_inc, dummy_grid_, dummy_grid_);
      auto absorption_vector = absorption_vector_.interpolate_angles(lon_inc, lat_inc, dummy_grid_, dummy_grid_);
      auto backward_scattering_coeff = backward_scattering_coeff_.interpolate_angles(lon_inc, lat_inc, dummy_grid_, dummy_grid_);
      auto forward_scattering_coeff = forward_scattering_coeff_.interpolate_angles(lon_inc, lat_inc, dummy_grid_, dummy_grid_);

      return new SingleScatteringDataGridded(f_grid_,
                                             t_grid_,
                                             phase_matrix,
                                             extinction_matrix,
                                             absorption_vector,
                                             backward_scattering_coeff,
                                             forward_scattering_coeff);
  }

 private:
  VectorPtr dummy_grid_ = std::make_shared<Vector>(Vector::Constant(1, 1));
  ScatteringDataFieldGridded<Scalar> phase_matrix_;
  ScatteringDataFieldGridded<Scalar> extinction_matrix_;
  ScatteringDataFieldGridded<Scalar> absorption_vector_;
  ScatteringDataFieldGridded<Scalar> backward_scattering_coeff_;
  ScatteringDataFieldGridded<Scalar> forward_scattering_coeff_;
};

////////////////////////////////////////////////////////////////////////////////
// Spectral single scattering data.
////////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
class SingleScatteringDataSpectral : public SingleScatteringDataBase<Scalar>,
                                     public SingleScatteringDataImpl {
  using SingleScatteringDataBase<Scalar>::f_grid_;
  using SingleScatteringDataBase<Scalar>::n_freqs_;
  using SingleScatteringDataBase<Scalar>::n_temps_;
  using SingleScatteringDataBase<Scalar>::t_grid_;

 public:
  using Vector = eigen::Vector<Scalar>;
  using VectorPtr = std::shared_ptr<Vector>;
  using ShtPtr = std::shared_ptr<sht::SHT>;
  using DataTensor = eigen::Tensor<std::complex<Scalar>, 6>;
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
      ScatteringDataFieldSpectral<Scalar> backward_scattering_coeff,
      ScatteringDataFieldSpectral<Scalar> forward_scattering_coeff)

      : SingleScatteringDataBase<Scalar>(f_grid, t_grid),
        phase_matrix_(phase_matrix),
        extinction_matrix_(extinction_matrix),
        absorption_vector_(absorption_vector),
        backward_scattering_coeff_(backward_scattering_coeff),
        forward_scattering_coeff_(forward_scattering_coeff) {}

  SingleScatteringDataSpectral(VectorPtr f_grid,
                               VectorPtr t_grid,
                               VectorPtr lon_inc,
                               VectorPtr lat_inc,
                               ShtPtr sht_scat,
                               DataPtr phase_matrix,
                               DataPtr extinction_matrix,
                               DataPtr absorption_vector,
                               DataPtr backward_scattering_coeff,
                               DataPtr forward_scattering_coeff)
      : SingleScatteringDataBase<Scalar>(f_grid, t_grid),
        sht_scat_(sht_scat),
        phase_matrix_{f_grid, t_grid, lon_inc, lat_inc, sht_scat, phase_matrix},
        extinction_matrix_{f_grid,
                           t_grid,
                           lon_inc,
                           lat_inc,
                           sht_dummy_,
                           extinction_matrix},
        absorption_vector_{f_grid,
                           t_grid,
                           lon_inc,
                           lat_inc,
                           sht_dummy_,
                           absorption_vector},
        backward_scattering_coeff_{f_grid,
                                   t_grid,
                                   lon_inc,
                                   lat_inc,
                                   sht_dummy_,
                                   backward_scattering_coeff},
        forward_scattering_coeff_{f_grid,
                                  t_grid,
                                  lon_inc,
                                  lat_inc,
                                  sht_dummy_,
                                  forward_scattering_coeff} {}

  eigen::Vector<double> get_f_grid() { return *f_grid_; }
  eigen::Vector<double> get_t_grid() { return *t_grid_; }
  eigen::Vector<double> get_lon_inc() { return phase_matrix_.get_lon_inc(); }
  eigen::Vector<double> get_lat_inc() { return phase_matrix_.get_lat_inc(); }
  eigen::Vector<double> get_lon_scat() { return phase_matrix_.get_lon_scat(); }
  eigen::Vector<double> get_lat_scat() { return phase_matrix_.get_lat_scat(); }

  eigen::Index get_n_lon_inc() { return phase_matrix_.get_n_lon_inc(); }
  eigen::Index get_n_lat_inc() { return phase_matrix_.get_n_lon_inc(); }
  eigen::Index get_n_lon_scat() { return phase_matrix_.get_n_lon_inc(); }
  eigen::Index get_n_lat_scat() { return phase_matrix_.get_n_lon_inc(); }

  void set_data(Index f_index,
                Index t_index,
                const SingleScatteringDataImpl &other) {
    auto converted = other.to_spectral();
    phase_matrix_.set_data(f_index, t_index, converted->phase_matrix_);
    extinction_matrix_.set_data(f_index,
                                t_index,
                                converted->extinction_matrix_);
    absorption_vector_.set_data(f_index,
                                t_index,
                                converted->absorption_vector_);
    backward_scattering_coeff_.set_data(f_index,
                                        t_index,
                                        converted->backward_scattering_coeff_);
    forward_scattering_coeff_.set_data(f_index,
                                       t_index,
                                       converted->forward_scattering_coeff_);
  }

  eigen::Tensor<Scalar, 7> get_phase_matrix() const {
    auto phase_matrix_gridded = phase_matrix_.to_gridded();
    return phase_matrix_gridded.get_data();
  }

  eigen::Tensor<std::complex<Scalar>, 6> get_phase_matrix_spectral() const {
    return phase_matrix_.get_data();
  }

  eigen::Tensor<Scalar, 7> get_extinction_matrix() const {
    auto extinction_matrix_gridded = extinction_matrix_.to_gridded();
    return extinction_matrix_gridded.get_data();
  }

  eigen::Tensor<Scalar, 7> get_absorption_vector() const {
    auto absorption_vector_gridded = absorption_vector_.to_gridded();
    return absorption_vector_gridded.get_data();
  }

  eigen::Tensor<Scalar, 7> get_backward_scattering_coeff() const {
    auto backward_scattering_coeff_gridded =
        backward_scattering_coeff_.to_gridded();
    return backward_scattering_coeff_gridded.get_data();
  }

  eigen::Tensor<Scalar, 7> get_forward_scattering_coeff() const {
    auto forward_scattering_coeff_gridded =
        forward_scattering_coeff_.to_gridded();
    return forward_scattering_coeff_gridded.get_data();
  }

  SingleScatteringDataSpectral *copy() {
    return new SingleScatteringDataSpectral(f_grid_,
                                            t_grid_,
                                            phase_matrix_.copy(),
                                            extinction_matrix_.copy(),
                                            absorption_vector_.copy(),
                                            backward_scattering_coeff_.copy(),
                                            forward_scattering_coeff_.copy());
  }

  SingleScatteringDataImpl *interpolate_frequency(
      eigen::VectorPtr<double> frequencies) {
    auto phase_matrix = ScatteringDataFieldSpectral<Scalar>(
        phase_matrix_.interpolate_frequency(frequencies));
    auto extinction_matrix = ScatteringDataFieldSpectral<Scalar>(
        extinction_matrix_.interpolate_frequency(frequencies));
    auto absorption_vector = ScatteringDataFieldSpectral<Scalar>(
        absorption_vector_.interpolate_frequency(frequencies));
    auto backward_scattering_coeff = ScatteringDataFieldSpectral<Scalar>(
        backward_scattering_coeff_.interpolate_frequency(frequencies));
    auto forward_scattering_coeff = ScatteringDataFieldSpectral<Scalar>(
        forward_scattering_coeff_.interpolate_frequency(frequencies));
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
    auto backward_scattering_coeff = ScatteringDataFieldSpectral<Scalar>(
        backward_scattering_coeff_.interpolate_temperature(temperatures));
    auto forward_scattering_coeff = ScatteringDataFieldSpectral<Scalar>(
        forward_scattering_coeff_.interpolate_temperature(temperatures));

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
    auto backward_scattering_coeff = ScatteringDataFieldSpectral<Scalar>(
        backward_scattering_coeff_.interpolate_angles(lon_inc, lat_inc));
    auto forward_scattering_coeff = ScatteringDataFieldSpectral<Scalar>(
        forward_scattering_coeff_.interpolate_angles(lon_inc, lat_inc));

    return new SingleScatteringDataSpectral(f_grid_,
                                            t_grid_,
                                            phase_matrix,
                                            extinction_matrix,
                                            absorption_vector,
                                            backward_scattering_coeff,
                                            forward_scattering_coeff);
  }

  void operator+=(const SingleScatteringDataImpl *other) {
    auto converted = other->to_spectral();
    phase_matrix_ += converted->phase_matrix_;
    extinction_matrix_ += converted->extinction_matrix_;
    absorption_vector_ += converted->absorption_vector_;
    backward_scattering_coeff_ += converted->backward_scattering_coeff_;
    forward_scattering_coeff_ += converted->forward_scattering_coeff_;
  }

  SingleScatteringDataImpl *operator+(const SingleScatteringDataImpl *other) {
    auto result = this->copy();
    result->operator+=(other);
    return result;
  }

  void operator*=(Scalar c) {
    phase_matrix_ *= c;
    extinction_matrix_ *= c;
    absorption_vector_ *= c;
    backward_scattering_coeff_ *= c;
    forward_scattering_coeff_ *= c;
  }

  SingleScatteringDataImpl *operator*(Scalar c) {
    auto result = this->copy();
    result->operator*=(c);
    return result;
  }

  void normalize(Scalar norm) { phase_matrix_.normalize(norm); }

  void set_number_of_scattering_coeffs(Index n) {
      phase_matrix_.set_number_of_scattering_coeffs(n);
      extinction_matrix_.set_number_of_scattering_coeffs(n);
      absorption_vector_.set_number_of_scattering_coeffs(n);
      backward_scattering_coeff_.set_number_of_scattering_coeffs(n);
      forward_scattering_coeff_.set_number_of_scattering_coeffs(n);
  }

  SingleScatteringDataImpl* regrid() {
      auto n_lon_inc = phase_matrix_.get_n_lon_inc();
      auto n_lat_inc = phase_matrix_.get_n_lat_inc();
      n_lon_inc = std::max<Index>(n_lon_inc - n_lon_inc % 2, 1);
      n_lat_inc = std::max<Index>(n_lat_inc - n_lat_inc % 2, 1);

      auto lon_inc = std::make_shared<Vector>(sht::SHT::get_longitude_grid(n_lon_inc));
      auto lat_inc = std::make_shared<Vector>(sht::SHT::get_latitude_grid(n_lat_inc));

      auto phase_matrix = phase_matrix_.interpolate_angles(lon_inc, lat_inc);
      auto extinction_matrix = extinction_matrix_.interpolate_angles(lon_inc, lat_inc);
      auto absorption_vector = absorption_vector_.interpolate_angles(lon_inc, lat_inc);
      auto backward_scattering_coeff = backward_scattering_coeff_.interpolate_angles(lon_inc, lat_inc);
      auto forward_scattering_coeff = forward_scattering_coeff_.interpolate_angles(lon_inc, lat_inc);

      return new SingleScatteringDataSpectral(f_grid_,
                                              t_grid_,
                                              phase_matrix,
                                              extinction_matrix,
                                              absorption_vector,
                                              backward_scattering_coeff,
                                              forward_scattering_coeff);
  }

  detail::ConversionPtr<const SingleScatteringDataGridded<Scalar>> to_gridded()
      const;

  detail::ConversionPtr<const SingleScatteringDataSpectral> to_spectral()
      const {
    return detail::make_conversion_ptr(this, false);
  }

  detail::ConversionPtr<const SingleScatteringDataSpectral> to_spectral(
      Index l_max,
      Index m_max) const;

  detail::ConversionPtr<const SingleScatteringDataSpectral> to_spectral(
      Index l_max,
      Index m_max,
      Index n_lat,
      Index n_lon) const;

  // explicit operator SingeScatteringDataSpectral();
  // explicit operator SingeScatteringDataFullySpectral();

 private:
  ShtPtr sht_scat_;
  ShtPtr sht_dummy_ = std::make_shared<sht::SHT>(0, 0, 1, 1);
  ScatteringDataFieldSpectral<Scalar> phase_matrix_;
  ScatteringDataFieldSpectral<Scalar> extinction_matrix_;
  ScatteringDataFieldSpectral<Scalar> absorption_vector_;
  ScatteringDataFieldSpectral<Scalar> backward_scattering_coeff_;
  ScatteringDataFieldSpectral<Scalar> forward_scattering_coeff_;
};

// pxx :: hide
template <typename Scalar>
class SingleScatteringDataFullySpectral {};

////////////////////////////////////////////////////////////////////////////////
// Member function definitions.
////////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
detail::ConversionPtr<const SingleScatteringDataSpectral<Scalar>>
SingleScatteringDataGridded<Scalar>::to_spectral(
    std::shared_ptr<sht::SHT> sht) const {
  using ReturnType = const SingleScatteringDataSpectral<Scalar>;
  auto phase_matrix =
      ScatteringDataFieldSpectral<Scalar>(phase_matrix_.to_spectral(sht));
  auto extinction_matrix =
      ScatteringDataFieldSpectral<Scalar>(extinction_matrix_.to_spectral());
  auto absorption_vector =
      ScatteringDataFieldSpectral<Scalar>(absorption_vector_.to_spectral());
  auto backward_scattering_coeff = ScatteringDataFieldSpectral<Scalar>(
      backward_scattering_coeff_.to_spectral());
  auto forward_scattering_coeff = ScatteringDataFieldSpectral<Scalar>(
      forward_scattering_coeff_.to_spectral());
  return detail::make_conversion_ptr<ReturnType>(
      new SingleScatteringDataSpectral<Scalar>(f_grid_,
                                               t_grid_,
                                               phase_matrix,
                                               extinction_matrix,
                                               absorption_vector,
                                               backward_scattering_coeff,
                                               forward_scattering_coeff),
      true);
}

template <typename Scalar>
detail::ConversionPtr<const SingleScatteringDataSpectral<Scalar>>
SingleScatteringDataGridded<Scalar>::to_spectral() const {
  auto sht_params = phase_matrix_.get_sht_scat_params();
  auto sht_scat = std::make_shared<sht::SHT>(sht_params[0],
                                             sht_params[1],
                                             phase_matrix_.get_n_lon_scat(),
                                             phase_matrix_.get_n_lat_scat());
  return to_spectral(sht_scat);
}

template <typename Scalar>
detail::ConversionPtr<const SingleScatteringDataSpectral<Scalar>>
SingleScatteringDataGridded<Scalar>::to_spectral(Index l_max,
                                                 Index m_max) const {
  auto sht_scat = std::make_shared<sht::SHT>(l_max,
                                             m_max,
                                             phase_matrix_.get_n_lon_scat(),
                                             phase_matrix_.get_n_lat_scat());
  return to_spectral(sht_scat);
}

template <typename Scalar>
    detail::ConversionPtr<const SingleScatteringDataSpectral<Scalar>>
    SingleScatteringDataGridded<Scalar>::to_spectral(Index l_max,
                                                     Index m_max,
                                                     Index n_lon,
                                                     Index n_lat) const {
    auto sht_scat = std::make_shared<sht::SHT>(l_max,
                                               m_max,
                                               n_lon,
                                               n_lat);
    return to_spectral(sht_scat);
}

template <typename Scalar>
detail::ConversionPtr<const SingleScatteringDataGridded<Scalar>>
SingleScatteringDataSpectral<Scalar>::to_gridded() const {
  using ReturnType = const SingleScatteringDataGridded<Scalar>;
  auto phase_matrix =
      ScatteringDataFieldGridded<Scalar>(phase_matrix_.to_gridded());
  auto extinction_matrix =
      ScatteringDataFieldGridded<Scalar>(extinction_matrix_.to_gridded());
  auto absorption_vector =
      ScatteringDataFieldGridded<Scalar>(absorption_vector_.to_gridded());
  auto backward_scattering_coeff = ScatteringDataFieldGridded<Scalar>(
      backward_scattering_coeff_.to_gridded());
  auto forward_scattering_coeff = ScatteringDataFieldGridded<Scalar>(
      forward_scattering_coeff_.to_gridded());
  auto lon_scat =
      std::make_shared<eigen::Vector<Scalar>>(phase_matrix_.get_lon_scat());
  auto lat_scat =
      std::make_shared<eigen::Vector<Scalar>>(phase_matrix_.get_lat_scat());
  return detail::make_conversion_ptr<ReturnType>(
      new SingleScatteringDataGridded<Scalar>(f_grid_,
                                              t_grid_,
                                              phase_matrix,
                                              extinction_matrix,
                                              absorption_vector,
                                              backward_scattering_coeff,
                                              forward_scattering_coeff),
      true);
}

template <typename Scalar>
detail::ConversionPtr<const SingleScatteringDataSpectral<Scalar>>
SingleScatteringDataSpectral<Scalar>::to_spectral(Index l_max,
                                                  Index m_max,
                                                  Index n_lon,
                                                  Index n_lat) const {
  using ReturnType = const SingleScatteringDataSpectral<Scalar>;
  auto phase_matrix = ScatteringDataFieldSpectral<Scalar>(
      phase_matrix_.to_spectral(l_max, m_max, n_lon, n_lat));
  auto extinction_matrix =
      ScatteringDataFieldSpectral<Scalar>(extinction_matrix_);
  auto absorption_vector =
      ScatteringDataFieldSpectral<Scalar>(absorption_vector_);
  auto backward_scattering_coeff =
      ScatteringDataFieldSpectral<Scalar>(backward_scattering_coeff_);
  auto forward_scattering_coeff =
      ScatteringDataFieldSpectral<Scalar>(forward_scattering_coeff_);
  return detail::make_conversion_ptr<ReturnType>(
      new SingleScatteringDataSpectral<Scalar>(f_grid_,
                                               t_grid_,
                                               phase_matrix,
                                               extinction_matrix,
                                               absorption_vector,
                                               backward_scattering_coeff,
                                               forward_scattering_coeff),
      true);
}

template <typename Scalar>
detail::ConversionPtr<const SingleScatteringDataSpectral<Scalar>>
SingleScatteringDataSpectral<Scalar>::to_spectral(Index l_max,
                                                  Index m_max) const {
  auto sht = phase_matrix_.get_sht_scat();
  auto n_lat = sht.get_n_latitudes();
  auto n_lon = sht.get_n_longitudes();
  return to_spectral(l_max, m_max, n_lon, n_lat);
}

////////////////////////////////////////////////////////////////////////////////
// SingleScatteringData
////////////////////////////////////////////////////////////////////////////////

SingleScatteringData SingleScatteringData::to_gridded() const {
  return SingleScatteringData(
      new SingleScatteringDataGridded<double>(*data_->to_gridded()));
}

SingleScatteringData SingleScatteringData::to_spectral() const {
  return SingleScatteringData(
      new SingleScatteringDataSpectral<double>(*data_->to_spectral()));
}

SingleScatteringData SingleScatteringData::to_spectral(Index l_max,
                                                       Index m_max) const {
  return SingleScatteringData(new SingleScatteringDataSpectral<double>(
      *data_->to_spectral(l_max, m_max)));
}

SingleScatteringData SingleScatteringData::to_spectral(Index l_max,
                                                       Index m_max,
                                                       Index n_lon,
                                                       Index n_lat) const {
    return SingleScatteringData(new SingleScatteringDataSpectral<double>(
                                    *data_->to_spectral(l_max, m_max, n_lon, n_lat)));
}

}  // namespace scatlib

#endif
