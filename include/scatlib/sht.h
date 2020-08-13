#ifndef __SCATLIB_SHT__
#define __SCATLIB_SHT__

#include <fftw3.h>
#include <shtns.h>
#include <scatlib/eigen.h>

#include <complex>
#include <iostream>
#include <memory>
#include <map>

#include "Eigen/Dense"

using GridCoeffs = scatlib::eigen::Matrix<double>;
using SpectralCoeffs = scatlib::eigen::Vector<std::complex<double>>;

namespace scatlib {
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
  FFTWArray(size_t n)
      : ptr_(reinterpret_cast<Numeric *>(fftw_malloc(n * sizeof(Numeric))),
             FFTWDeleter()) {}

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
  using Vector = eigen::Vector<double>;
  using ConstVectorMap = eigen::ConstVectorMap<double>;
  /**
   * Create a spherical harmonics transformation object.
   *
   * @param l_max The maximum degree of the SHT.
   * @param m_max The maximum order of the SHT.
   * @param n_lat The number of co-latitude grid points.
   * @param n_lon The number of longitude grid points.
   * @param m_res The order-resolution of the SHT.
   */
  SHT(size_t l_max, size_t m_max, size_t n_lat, size_t n_lon, size_t m_res = 1)
      : l_max_(l_max),
        m_max_(m_max),
        n_lat_(n_lat),
        n_lon_(n_lon),
        m_res_(m_res) {
    shtns_verbose(1);
    shtns_use_threads(0);
    shtns_reset();
    shtns_ = shtns_init(sht_quick_init, l_max_, m_max_, m_res_, n_lat_, n_lon_);
    spectral_coeffs_ = sht::FFTWArray<std::complex<double>>(shtns_->nlm);
    spatial_coeffs_ = sht::FFTWArray<double>(NSPAT_ALLOC(shtns_));
  }

  /** Return latitude grid used by SHTns.
   * @return Eigen vector containing the latitude grid in radians.
   */
  Vector get_latitude_grid() {
      return ConstVectorMap(shtns_->ct, n_lat_).array().acos();
  }

  /** Return co-latitude grid used by SHTns.
   * @return Eigen vector containing the co-latitude grid.
   */
  Vector get_colatitude_grid() {
      return ConstVectorMap(shtns_->ct, n_lat_);
  }

  /**
   * Copy spatial field into the array that holds spatial data for
   * spherical harmonics computations.
   *
   * @param m GridCoeffs containing the data. Row indices should correspond to
   * longitudes (azimuth angle) and columns to latitudes (zenith angle).
   */
  void set_spatial_coeffs(const GridCoeffs &m) const {
    size_t index = 0;
    for (int i = 0; i < m.rows(); ++i) {
      for (int j = 0; j < m.cols(); ++j) {
        spatial_coeffs_[index] = m(i, j);
        ++index;
      }
    }
  }

  /**
   * Copy spherical harmonics coefficients into the array that holds spectral
   * data for spherical harmonics computations.
   *
   * @param m SpectralCoeffs The spherical harmonics coefficients
   * representing the data.
   */
  void set_spectral_coeffs(const SpectralCoeffs &m) const {
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
   * @return m GridCoeffs containing the data. Row indices should correspond to
   * longitudes (azimuth angle) and columns to latitudes (zenith angle).
   */
  GridCoeffs get_spatial_coeffs() const {
      GridCoeffs result(n_lon_, n_lat_);
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
   * @return The size of the co-latitude grid.
   */
  size_t get_n_latitudes() const {
      return n_lat_;
  }
  /**
   * @return The size of the longitude grid.
   */
  size_t get_n_longitudes() const {
      return n_lon_;
  }
  /**
   * @return The number of spherical harmonics coefficients.
   */
  size_t get_number_of_spectral_coeffs() const {
    return shtns_->nlm;
  }

  /**
   * Return content of the array that holds spectral data for
   * spherical harmonics computations.
   *
   * @return m SpectralCoeffs The spherical harmonics coefficients
   * representing the data.
   */
  SpectralCoeffs get_spectral_coeffs() const {
    SpectralCoeffs result(shtns_->nlm);
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
   * @param m GridCoeffs containing the data. Row indices should correspond to
   * longitudes (azimuth angle) and columns to latitudes (zenith angle).
   * @return Coefficient vector containing the spherical harmonics coefficients.
   */
  SpectralCoeffs transform(const GridCoeffs &m) {
    set_spatial_coeffs(m);
    spat_to_SH(shtns_, spatial_coeffs_, spectral_coeffs_);
    return get_spectral_coeffs();
  }

  /** Apply inverse SHT Transform
   *
   * Transforms discrete spherical data given in spherical harmonics
   * representation back to spatial domain.
   *
   * @param m SpectralCoeffs The spherical harmonics coefficients
   * representing the data.
   * @return GridCoeffs containing the spatial data.
   */
  GridCoeffs transform(const SpectralCoeffs &m) {
    set_spectral_coeffs(m);
    SH_to_spat(shtns_, spectral_coeffs_, spatial_coeffs_);
    return get_spatial_coeffs();
  }

  /** Evaluate spectral representation at given point.
   *
   * @param m Spectral coefficient vector containing the SH coefficients.
   * @param points 2-row matrix containing the points (lon, lat) at which
   * to evaluate the function.
   * @return A vector containing the values corresponding to the points
   * in points.
   */
  eigen::Vector<double> evaluate(const SpectralCoeffs &m,
                                 const eigen::MatrixFixedRows<double, 2> &points) {
    set_spectral_coeffs(m);
    int n_points = points.rows();
    eigen::Vector<double> result(n_points);
    for (int i = 0; i < n_points; ++i) {
      result[i] = SH_to_point(shtns_,
                              spectral_coeffs_,
                              cos(points(i, 1)),
                              points(i, 0));
    }
    return result;
  }

  /** Evaluate 1D spectral representation at given point.
   *
   * This method covers the special case of 1D data that varies
   * only along latitudes. In this case the SH transform degenerates
   * to a Legendre transform.
   *
   * @param m Spectral coefficient vector containing the SH coefficients.
   * @param Vector containing the latitudes within [0, PI] to evaluate the function.
   * @return A vector containing the values corresponding to the points
   * in points.
   */
  eigen::Vector<double> evaluate(const SpectralCoeffs &m,
                                 const eigen::Vector<double> &thetas) {
      assert(m_max_ == 0);
      set_spectral_coeffs(m);
      int n_points = thetas.size();
      eigen::Vector<double> result(n_points);
      for (int i = 0; i < n_points; ++i) {
          result[i] = SH_to_point(shtns_,
                                  spectral_coeffs_,
                                  cos(thetas[i]),
                                  0.0);
      }
      return result;
  }

 private:
  size_t l_max_, m_max_, n_lat_, n_lon_, m_res_;
  shtns_cfg shtns_;

  sht::FFTWArray<std::complex<double>> spectral_coeffs_;
  sht::FFTWArray<double> spatial_coeffs_;
};

/** SHT instance provider.
 *
 * Simple cache that caches created SHT instances.
 */
class SHTProvider {
public:
    using SHTParams = std::array<size_t, 4>;

    SHTProvider();

    /** Get SHT instance for given SHT parameters.
     * @arg params Lenght-4 array containing the parameters required to initialize the SHT
     * transform: l_max, m_max, n_lat, n_lon. See documention of SHT class for explanation
     * of their significance.
     * @return Reference to SHT instance.
     */
    SHT& get_sht_instance(SHTParams params) {
        if (sht_instances_.count(params) == 0) {
            sht_instances_[params] = std::make_unique<SHT>(params[0], params[1], params[2], params[3]);
        }
        return *sht_instances_[params];
    }

protected:
    std::map<SHTParams, std::unique_ptr<SHT>> sht_instances_;
};

}  // namespace sht
}  // namespace scatlib

#endif
