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
using GridCoeffsRef = scatlib::eigen::ConstMatrixRef<double>;
using CmplxGridCoeffs = scatlib::eigen::Matrix<std::complex<double>>;
using CmplxGridCoeffsRef = scatlib::eigen::ConstMatrixRef<std::complex<double>>;
using SpectralCoeffs = scatlib::eigen::Vector<std::complex<double>>;
using SpectralCoeffsRef = scatlib::eigen::ConstVectorRef<std::complex<double>>;
using SpectralCoeffMatrix = scatlib::eigen::Matrix<std::complex<double>>;
using SpectralCoeffMatrixRef = scatlib::eigen::ConstMatrixRef<std::complex<double>>;

namespace scatlib {
namespace sht {

using scatlib::eigen::Index;

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
  FFTWArray(Index n)
      : ptr_(reinterpret_cast<Numeric *>(fftw_malloc(n * sizeof(Numeric))),
             FFTWDeleter()) {}

  operator Numeric *() const { return ptr_.get(); }

 private:
  std::shared_ptr<Numeric> ptr_ = nullptr;
};

class ShtnsHandle {
 public:
  static shtns_cfg get(Index l_max, Index m_max, Index n_lat, Index n_lon) {
    std::array<Index, 4> config = {l_max, m_max, n_lat, n_lon};
    if (config == current_config_) {
      return shtns_;
    } else {
      shtns_reset();
      shtns_ = shtns_init(sht_quick_init, l_max, m_max, 1, n_lat, n_lon);
      current_config_ = config;
    }
    return shtns_;
  }

 private:
  static std::array<Index, 4> current_config_;
  static shtns_cfg shtns_;
};

shtns_cfg ShtnsHandle::shtns_ = nullptr;
std::array<Index, 4> ShtnsHandle::current_config_ = {-1, -1, -1, -1};

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
  using IndexVector = eigen::Vector<Index>;
  using ConstVectorMap = eigen::ConstVectorMap<double>;

  static SpectralCoeffs add_coeffs(const SHT &sht_l,
                                   SpectralCoeffsRef v,
                                   const SHT &sht_r,
                                   SpectralCoeffsRef w) {
    auto result = SpectralCoeffs(v);

    if (sht_r.is_trivial_) {
        result[0] += w[0];
        return result;
    }

    Index m_max_min = std::min(sht_l.m_max_, sht_r.m_max_);
    Index l_max_min = std::min(sht_l.l_max_, sht_r.l_max_);
    for (Index m = 0; m <= m_max_min; ++m) {
        Index index_r = m * (sht_r.l_max_ + 1) - (m * (m - 1)) / 2;
        Index index_l = m * (sht_l.l_max_ + 1) - (m * (m - 1)) / 2;
        for (Index l = m; l <= l_max_min; ++l) {
            result[index_l] += w[index_r];
            ++index_r;
            ++index_l;
        }
    }
    return result;
  }

    static SpectralCoeffMatrix add_coeffs(const SHT &sht_inc_l,
                                          const SHT &sht_scat_l,
                                          SpectralCoeffMatrixRef v,
                                          const SHT &sht_inc_r,
                                          const SHT &sht_scat_r,
                                          SpectralCoeffMatrixRef w) {
      Index nlm_inc = sht_inc_l.get_n_spectral_coeffs_cmplx();
      Index nlm_scat = sht_scat_l.get_n_spectral_coeffs();
      auto result = SpectralCoeffMatrix(nlm_inc, nlm_scat);

      Index index_l = 0;
      for (int l = 0; l <= (int)sht_inc_l.l_max_; ++l) {
        int m_max = (l <= (int)sht_inc_l.m_max_) ? l : sht_inc_l.m_max_;
        for (int m = -m_max; m <= m_max; ++m) {
          if ((l > sht_inc_r.l_max_) || (std::abs(m) > sht_inc_r.m_max_)) {
            result.row(index_l) = v.row(index_l);
        } else {
          int h = std::min<int>(sht_inc_r.m_max_, l);
          int index_r = l * (2 * h + 1) - h * h + m;

          auto r = add_coeffs(sht_scat_l,
                              v.row(index_l),
                              sht_scat_r,
                              w.row(index_r));
          result.row(index_l) = r;
        }
        ++index_l;
      }
    }
    return result;
  }

  /** Calculates the number of spherical harmonics coefficients for a real transform.
   * @param l_max The maximum degree of the SHT.
   * @param m_max The maximum order of the SHT.
   * @return The number of spherical harmonics coefficients.
   */
  static Index calc_n_spectral_coeffs(Index l_max, Index m_max) {
    return (l_max + 1) * (m_max + 1) - (m_max * (m_max + 1)) / 2;
  }

  /** Calculates the number of spherical harmonics coefficients for a complex transform.
   * @param l_max The maximum degree of the SHT.
   * @param m_max The maximum order of the SHT.
   * @return The number of spherical harmonics coefficients for a complex transform.
   */
  static Index calc_n_spectral_coeffs_cmplx(Index l_max, Index m_max) {
    return (2 * m_max + 1) * (l_max + 1) - m_max * (m_max + 1);
  }

  /** Calc l_max for m_max == l_max.
   *
   * Calculates the value of l_max that yields the given number of spectral
   * coefficients under the assumption that m_max is equal to l_max.
   *
   * @param n_spectral_coeffs The number of spectral coefficients.
   * @return l_max value yielding the given number of spectral coeffs.
   */
  static Index calc_l_max(Index n_spectral_coeffs) {
      return static_cast<Index>(sqrt(2 * (n_spectral_coeffs - 1) + 2.25) - 1.5);
  }

  /**
   * Create a spherical harmonics transformation object.
   *
   * @param l_max The maximum degree of the SHT.
   * @param m_max The maximum order of the SHT.
   * @param n_lat The number of co-latitude grid points.
   * @param n_lon The number of longitude grid points.
   */
  SHT(Index l_max, Index m_max, Index n_lat, Index n_lon)
      : l_max_(l_max),
        m_max_(m_max),
        n_lat_(n_lat),
        n_lon_(n_lon) {
    if (l_max == 0) {
      is_trivial_ = true;
      n_spectral_coeffs_ = 1;
      n_spectral_coeffs_cmplx_ = 1;
    } else {
      is_trivial_ = false;
      shtns_verbose(1);
      shtns_use_threads(0);
      n_spectral_coeffs_ = calc_n_spectral_coeffs(l_max, m_max);
      n_spectral_coeffs_cmplx_ = calc_n_spectral_coeffs_cmplx(l_max, m_max);
      spectral_coeffs_ =
          sht::FFTWArray<std::complex<double>>(n_spectral_coeffs_);
      spectral_coeffs_cmplx_ =
          sht::FFTWArray<std::complex<double>>(n_spectral_coeffs_cmplx_);
      spatial_coeffs_ = sht::FFTWArray<double>(n_lon * n_lat);
      cmplx_spatial_coeffs_ =
          sht::FFTWArray<std::complex<double>>(n_lon * n_lat);
    }
  }

  /** Return latitude grid used by SHTns.
   * @return Eigen vector containing the latitude grid in radians.
   */
  Vector get_latitude_grid() {
      if (is_trivial_) {
          return Vector::Constant(1, M_PI / 2.0);
      }
      auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lat_, n_lon_);
      return ConstVectorMap(shtns->ct, n_lat_).array().acos();
  }

  /** Return co-latitude grid used by SHTns.
   * @return Eigen vector containing the co-latitude grid.
   */
  Vector get_colatitude_grid() {
      if (is_trivial_) {
          return Vector::Constant(1, M_PI / 2.0);
      }
      auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lat_, n_lon_);
      return ConstVectorMap(shtns->ct, n_lat_);
  }

  Vector get_longitude_grid() {
      Vector v{n_lon_};
      double dx = 2 * M_PI / (n_lon_ + 1);
      for (Index i = 0; i < n_lon_; ++i) {
          v[i] = dx * i;
      }
      return v;
  }

  /** L-indices of the SHT modes.
   *
   * @return A vector of indices containing the l-value corresponding to each
   * element in a spectral coefficient vector.
   */
  IndexVector get_l_indices() {
      if (is_trivial_) {
          return IndexVector::Constant(1, 0);
      }
    auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lat_, n_lon_);
    IndexVector result(n_spectral_coeffs_);
    for (Index i = 0; i < n_spectral_coeffs_; ++i) {
      result[i] = shtns->li[i];
    }
    return result;
  }

  /** M-indices of the SHT modes.
   *
   * @return A vector of indices containing the m-value corresponding to each
   * element in a spectral coefficient vector.
   */
  IndexVector get_m_indices() {
      if (is_trivial_) {
          return IndexVector::Constant(1, 0);
      }
    auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lat_, n_lon_);
    IndexVector result(n_spectral_coeffs_);
    for (Index i = 0; i < n_spectral_coeffs_; ++i) {
      result[i] = shtns->mi[i];
    }
    return result;
  }

  /**
   * Copy spatial field into the array that holds spatial data for
   * spherical harmonics computations.
   *
   * @param m Eigen::Matrix or comparable providing read-only access
   * to the input data. Row indices should correspond to longitudes
   * (azimuth angle) and columns to latitudes (zenith angle).
   */
  void set_spatial_coeffs(const GridCoeffsRef &m) const {
    Index index = 0;
    for (int i = 0; i < m.rows(); ++i) {
      for (int j = 0; j < m.cols(); ++j) {
        spatial_coeffs_[index] = m(i, j);
        ++index;
      }
    }
  }

  /**
   * Copy complex spatial field into the array that holds spatial data for
   * spherical harmonics computations.
   *
   * @param m Eigen::Matrix or comparable providing read-only access
   * to the input data. Row indices should correspond to longitudes
   * (azimuth angle) and columns to latitudes (zenith angle).
   */
  void set_spatial_coeffs(const CmplxGridCoeffsRef &m) const {
    Index index = 0;
    for (int i = 0; i < m.rows(); ++i) {
      for (int j = 0; j < m.cols(); ++j) {
        cmplx_spatial_coeffs_[index] = m(i, j);
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
  void set_spectral_coeffs(const SpectralCoeffsRef &m) const {
    Index index = 0;
    for (auto &x : m) {
      spectral_coeffs_[index] = x;
      ++index;
    }
  }

  /**
   * Copy spherical harmonics coefficients into the array that holds spectral
   * data for spherical harmonics computations involving complex spatial fields.
   *
   * @param m Eigen vector containing the spherical harmonics coefficients
   * representing the data.
   */
  void set_spectral_coeffs_cmplx(const SpectralCoeffsRef &m) const {
      Index index = 0;
      for (auto &x : m) {
          spectral_coeffs_cmplx_[index] = x;
          ++index;
      }
  }

  /**
   * Return content of the array that holds spatial data for
   * spherical harmonics computations.
   *
   * @return Eigen matrix containing the spatial field. Row indices should
   * correspond to longitudes (azimuth angle) and columns to latitudes (zenith
   * angle).
   */
  GridCoeffs get_spatial_coeffs() const {
    GridCoeffs result(n_lon_, n_lat_);
    Index index = 0;
    for (int i = 0; i < result.rows(); ++i) {
      for (int j = 0; j < result.cols(); ++j) {
        result(i, j) = spatial_coeffs_[index];
        ++index;
      }
    }
    return result;
  }

  /**
   * Return content of the array that holds complex spatial data for
   * spherical harmonics computations.
   *
   * @return Eigen matrix containing the complex spatial field. Row indices
   * should correspond to longitudes (azimuth angle) and columns to latitudes
   * (zenith angle).
   */
  CmplxGridCoeffs get_cmplx_spatial_coeffs() const {
      CmplxGridCoeffs result(n_lon_, n_lat_);
      Index index = 0;
      for (int i = 0; i < result.rows(); ++i) {
          for (int j = 0; j < result.cols(); ++j) {
              result(i, j) = cmplx_spatial_coeffs_[index];
              ++index;
          }
      }
      return result;
  }

  /**
   * @return The size of the co-latitude grid.
   */
  Index get_n_latitudes() const {
      return n_lat_;
  }
  /**
   * @return The size of the longitude grid.
   */
  Index get_n_longitudes() const {
      return n_lon_;
  }
  /**
   * @return The number of spherical harmonics coefficients.
   */
  Index get_n_spectral_coeffs() const {
    return n_spectral_coeffs_;
  }
  /**
   * @return The number of spherical harmonics coefficients.
   */
  Index get_n_spectral_coeffs_cmplx() const {
      return n_spectral_coeffs_cmplx_;
  }

  /**
   * Return content of the array that holds spectral data for
   * spherical harmonics computations.
   *
   * @return m Eigen vector containing the spherical harmonics coefficients
   * representing the data.
   */
  SpectralCoeffs get_spectral_coeffs() const {
    SpectralCoeffs result(n_spectral_coeffs_);
    Index index = 0;
    for (auto &x : result) {
      x = spectral_coeffs_[index];
      ++index;
    }
    return result;
  }

  /**
   * Return content of the array that holds spectral data for
   * spherical harmonics computations of complex fields.
   *
   * @return m Eigen vector containing the spherical harmonics coefficients
   * representing the data.
   */
  SpectralCoeffs get_spectral_coeffs_cmplx() const {
      SpectralCoeffs result(n_spectral_coeffs_cmplx_);
      Index index = 0;
      for (auto &x : result) {
          x = spectral_coeffs_cmplx_[index];
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
  SpectralCoeffs transform(const GridCoeffsRef &m) {
    if (is_trivial_) {
      return SpectralCoeffs::Constant(1, m(0, 0));
    }
    set_spatial_coeffs(m);
    auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lat_, n_lon_);
    spat_to_SH(shtns, spatial_coeffs_, spectral_coeffs_);
    return get_spectral_coeffs();
  }

  /** Apply forward SHT Transform
   *
   * Transforms discrete spherical data into spherical harmonics representation.
   * @param m GridCoeffs containing the data. Row indices should correspond to
   * longitudes (azimuth angle) and columns to latitudes (zenith angle).
   * @return Coefficient vector containing the spherical harmonics coefficients.
   */
  SpectralCoeffs transform_cmplx(const CmplxGridCoeffsRef &m) {
      if (is_trivial_) {
          return SpectralCoeffs::Constant(1, m(0, 0));
      }
      set_spatial_coeffs(m);
      auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lat_, n_lon_);
      spat_cplx_to_SH(shtns, cmplx_spatial_coeffs_, spectral_coeffs_cmplx_);
      return get_spectral_coeffs_cmplx();
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
  GridCoeffs synthesize(const SpectralCoeffsRef &m) {
      if (is_trivial_) {
          return GridCoeffs::Constant(1, 1, m(0, 0).real());
      }
    set_spectral_coeffs(m);
    auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lat_, n_lon_);
    SH_to_spat(shtns, spectral_coeffs_, spatial_coeffs_);
    return get_spatial_coeffs();
  }

  /** Apply inverse SHT Transform for complex data.
   *
   * Transforms discrete spherical data given in spherical harmonics
   * representation back to spatial domain.
   *
   * @param m SpectralCoeffs The spherical harmonics coefficients
   * representing the data.
   * @return GridCoeffs containing the spatial data.
   */
  CmplxGridCoeffs synthesize_cmplx(const SpectralCoeffsRef &m) {
      if (is_trivial_) {
          return CmplxGridCoeffs::Constant(1, 1, m(0, 0).real());
      }
      set_spectral_coeffs_cmplx(m);
      auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lat_, n_lon_);
      SH_to_spat_cplx(shtns, spectral_coeffs_cmplx_, cmplx_spatial_coeffs_);
      return get_cmplx_spatial_coeffs();
  }

  /** Evaluate spectral representation at given point.
   *
   * @param m Spectral coefficient vector containing the SH coefficients.
   * @param points 2-row matrix containing the points (lon, lat) at which
   * to evaluate the function.
   * @return A vector containing the values corresponding to the points
   * in points.
   */
  eigen::Vector<double> evaluate(const SpectralCoeffsRef &m,
                                 const eigen::MatrixFixedRows<double, 2> &points) {
      if (is_trivial_) {
          return eigen::Vector<double>::Constant(1, 1, m(0, 0).real());
      }
    set_spectral_coeffs(m);
    int n_points = points.rows();
    eigen::Vector<double> result(n_points);
    auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lat_, n_lon_);
    for (int i = 0; i < n_points; ++i) {
      result[i] = SH_to_point(shtns,
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
  eigen::Vector<double> evaluate(const SpectralCoeffsRef &m,
                                 const eigen::Vector<double> &thetas) {
      if (is_trivial_) {
          return eigen::Vector<double>::Constant(1, 1, m(0, 0).real());
      }
      assert(m_max_ == 0);
      set_spectral_coeffs(m);
      int n_points = thetas.size();
      eigen::Vector<double> result(n_points);
      auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lat_, n_lon_);
      for (int i = 0; i < n_points; ++i) {
          result[i] = SH_to_point(shtns,
                                  spectral_coeffs_,
                                  cos(thetas[i]),
                                  0.0);
      }
      return result;
  }

 private:

  bool is_trivial_;
  Index l_max_, m_max_, n_lat_, n_lon_,
      n_spectral_coeffs_, n_spectral_coeffs_cmplx_;

  sht::FFTWArray<std::complex<double>> spectral_coeffs_,
      spectral_coeffs_cmplx_, cmplx_spatial_coeffs_;
  sht::FFTWArray<double> spatial_coeffs_;
  };

/** SHT instance provider.
 *
 * Simple cache that caches created SHT instances.
 */
class SHTProvider {
public:
    using SHTParams = std::array<Index, 4>;

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
