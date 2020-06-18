#ifndef __SCATLIB_SHT__
#define __SCATLIB_SHT__

#include <fftw3.h>
#include <shtns.h>
#include <complex>
#include <memory>
#include <iostream>
#include "Eigen/Dense"

namespace scatlib {

    using Matrix = Eigen::Matrix<double, -1, -1>;
    using CoefficientVector = Eigen::Matrix<std::complex<double>, -1, 1>;

namespace sht {

/** Deleter functional for smart pointers. */

struct FFTWDeleter {
  template <typename T>
  void operator()(T *t) {
    if (t) {
      fftw_free(t);
    }
  }
};

/** FFTW coefficient array
 *
 * SHTns library requires memory to be aligned according to fftw3
 * requirements which is ensure by using fftw_malloc and fftw_free
 * functions to allocate and free memory, respectively.
 */
template <typename Numeric>
class FFTWArray {

public:
    FFTWArray() {}
FFTWArray(size_t n) : ptr_(reinterpret_cast<Numeric*>(fftw_malloc(n * sizeof(Numeric))), FFTWDeleter()) {}

  operator Numeric *() const { return ptr_.get(); }

 private:
  std::shared_ptr<Numeric> ptr_ = nullptr;
};

////////////////////////////////////////////////////////////////////////////////
// SHT
////////////////////////////////////////////////////////////////////////////////
/** Spherical harmonics transformation
 *
 * Represents a spherical harmonics transformation (SHT) for fixed spatial- and
 * spectral grid sizes. A SHT object essentially acts as a wrapper around the
 * SHTns library. Each object holds a SHTns configuration as well as the array
 * required to store spatial and spectral coefficients.
 */
// pxx :: export
class SHT {
 public:
  /**
   * Create a spherical harmonics transformation object.
   *
   * @param l_max The maximum degree of the SHT.
   * @param m_max The maximum order of the SHT.
   * @param n_lat The number of co-latitude grid points.
   * @param n_phi The number of longitude grid points.
   * @param m_res The order-resolution of the SHT.
   */
  SHT(size_t l_max, size_t m_max, size_t n_lat, size_t n_phi, size_t m_res = 1)
      : l_max_(l_max),
        m_max_(m_max),
        n_lat_(n_lat),
        n_lon_(n_phi),
        m_res_(m_res) {
    shtns_verbose(1);
    shtns_use_threads(0);
    shtns_reset();
    shtns_ = shtns_init(sht_gauss, l_max_, m_max_, m_res_, n_lat_, n_lon_);
    spectral_coeffs_ = sht::FFTWArray<std::complex<double>>(shtns_->nlm);
    spatial_coeffs_ = sht::FFTWArray<double>(NSPAT_ALLOC(shtns_));
  }

  /**
   * Copy spatial field into the array that holds spatial data for
   * spherical harmonics computations.
   *
   * @param m Matrix containing the data. Row indices should correspond to
   * longitudes (azimuth angle) and columns to latitudes (zenith angle).
   */
  void set_spatial_coefficients(const Matrix &m) const {
      size_t index = 0;
      for (int i = 0; i < m.rows(); ++i) {
          for (int j = 0; j < m.cols(); ++j) {
              spatial_coeffs_[index] = m(i, j);
              ++index;
          }
      }
  }

  /**
   * Copy spherical harmonics coefficients into the array that holds spectral data for
   * spherical harmonics computations.
   *
   * @param m CoefficientVector The spherical harmonics coefficients representing the data.
   */
  void set_spectral_coefficients(const CoefficientVector &m) const {
      size_t index = 0;
      for (auto &x : m) {
          spectral_coeffs_[index] = x;
          ++index;
      }
  }

  /**
   * Return content of the array that holds spatial data for
   * spherical harmonics computations.
   *
   * @return m Matrix containing the data. Row indices should correspond to
   * longitudes (azimuth angle) and columns to latitudes (zenith angle).
   */
  Matrix get_spatial_coefficients() const {
      Matrix result(n_lon_, n_lat_);
      size_t index = 0;
      for (int i = 0; i < result.rows(); ++i) {
          for (int j = 0; j < result.cols(); ++j) {
              result(i, j) = spatial_coeffs_[index];
              ++index;
          }
      }
      return result;
  }

  /**
   * Return content of the array that holds spectral data for
   * spherical harmonics computations.
   *
   * @return m CoefficientVector The spherical harmonics coefficients representing the data.
   */
  CoefficientVector get_spectral_coefficients() const {
      CoefficientVector result(shtns_->nlm);
      size_t index = 0;
      for (auto &x : result) {
          x = spectral_coeffs_[index];
          ++index;
      }
      return result;
  }

/** Apply forward SHT Transform
  *
  * Transforms discrete spherical data into spherical harmonics representation.
  * @param m Matrix containing the data. Row indices should correspond to
  * longitudes (azimuth angle) and columns to latitudes (zenith angle).
  * @return Coefficient vector containing the spherical harmonics coefficients.
  */
  CoefficientVector transform(Matrix m) {
    set_spatial_coefficients(m);
    spat_to_SH(shtns_, spatial_coeffs_, spectral_coeffs_);
    return get_spectral_coefficients();
  }

  /** Apply inverse SHT Transform
  *
  * Transforms discrete spherical data given in spherical harmonics representation
  * back to spatial domain.
  *
  * @param m CoefficientVector The spherical harmonics coefficients representing the data.
  * @return Matrix containing the spatial data.
  */
  Matrix transform(CoefficientVector m) {
    set_spectral_coefficients(m);
    SH_to_spat(shtns_, spectral_coeffs_, spatial_coeffs_);
    return get_spatial_coefficients();
  }

 private:
  size_t l_max_, m_max_, n_lat_, n_lon_, m_res_;
  shtns_cfg shtns_;

  sht::FFTWArray<std::complex<double>> spectral_coeffs_;
  sht::FFTWArray<double> spatial_coeffs_;
};
}
}  // namespace scatlib

#endif
