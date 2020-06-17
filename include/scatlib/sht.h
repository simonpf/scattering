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

namespace detail {

template <typename T>
void copy(const T &m, typename T::Scalar *ptr) {
  size_t index = 0;
  for (auto &&x : m.reshaped()) {
    ptr[index] = x;
    index++;
  }
}

template <typename T>
void copy(const typename T::Scalar *ptr, T &m) {
  size_t index = 0;
  for (auto &&x : m.reshaped()) {
    x = ptr[index];
    index++;
  }
}
}  // namespace detail


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

//template <typename Numeric>
//class SpectralData {
//public:
//SpectralData(size_t l_max, size_t m_max, size_t m_res = 1) :
//    l_max_(l_max), m_max_(m_max), m_res_(m_res)
//
//
//private:
//    size_t l_max_, m_max_, m_res_;
//
//
//};
//
//template <typename Numeric>
//class SphericalData {
//public:
//
//    SphericalData(size_t n_lat, n_phi)
//        : n_lat_(n_lat), n_phi_(n_phi), data_(n_lat * n_phi) {
//        // Nothing to do here.
//    }
//
//
//private:
//    size_T n_lat_, n_phi_;
//    CoefficientArray<Numeric> data_;
//};


using detail::copy;

/** Spherical harmonics transformation
 *
 * Represents a spherical harmonics transformation (SHT) for fixed spatial- and spectral
 * grid sizes. A SHT object essentially acts as a wrapper around the SHTns library.
 * Each object holds a SHTns configuration as well as the array required to store
 * spatial and spectral coefficients.
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
        m_res_(m_res)
    {
        shtns_verbose(1);
        shtns_use_threads(0);
        shtns_reset();
        shtns_ = shtns_init(sht_gauss, l_max_, m_max_, m_res_, n_lat_, n_lon_);
        spectral_coeffs_ = sht::FFTWArray<std::complex<double>>(shtns_->nlm);
        spatial_coeffs_ = sht::FFTWArray<double>(NSPAT_ALLOC(shtns_));
    }

    CoefficientVector transform(Matrix m) {

        copy(m, spatial_coeffs_);
        spat_to_SH(shtns_, spatial_coeffs_, spectral_coeffs_);
        CoefficientVector results(shtns_->nlm);
        copy(spectral_coeffs_, results);
        return results;
    }

    Matrix get_spatial_coefficients() const {
      Matrix result(n_lat_, n_lon_);
      size_t index = 0;
        for (size_t i = 0; i < result.rows(); ++i) {
          for (size_t j = 0; j < result.cols(); ++j) {
              result(i, j) = spatial_coeffs_[index];
              std::cout << spatial_coeffs_[index] << std::endl;
              ++index;
          }
        }
        return result;
      }

    CoefficientVector get_spectral_coefficients() const {
        CoefficientVector result(shtns_->nlm);
        for (size_t i = 0; i < shtns_->nlm; ++i) {
            result(i) = spectral_coeffs_[i];
        }
        return result;
    }

    Matrix transform(CoefficientVector m) {
        copy(m, spectral_coeffs_);
        SH_to_spat(shtns_, spectral_coeffs_, spatial_coeffs_);
        Matrix results(n_lon_, n_lat_);
        copy(spatial_coeffs_, results);
        return results;
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
