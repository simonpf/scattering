#ifndef __SCATTERING_SHT__
#define __SCATTERING_SHT__

#include <fftw3.h>
#include <shtns.h>

#include <complex>
#include <iostream>
#include <map>
#include <memory>

#include "Eigen/Dense"

#include <scattering/eigen.h>
#include <scattering/integration.h>

using GridCoeffs = scattering::eigen::Matrix<double>;
using GridCoeffsRef = scattering::eigen::ConstMatrixRef<double>;
using CmplxGridCoeffs = scattering::eigen::Matrix<std::complex<double>>;
using CmplxGridCoeffsRef = scattering::eigen::ConstMatrixRef<std::complex<double>>;
using SpectralCoeffs = scattering::eigen::Vector<std::complex<double>>;
using SpectralCoeffsRef = scattering::eigen::ConstVectorRef<std::complex<double>>;
using SpectralCoeffMatrix = scattering::eigen::Matrix<std::complex<double>>;
using SpectralCoeffMatrixRef =
    scattering::eigen::ConstMatrixRef<std::complex<double>>;

namespace scattering {
namespace sht {

using scattering::eigen::Index;


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
  FFTWArray(Index n);
  operator Numeric *() const { return ptr_.get(); }

 private:
  std::shared_ptr<Numeric> ptr_ = nullptr;
};

class ShtnsHandle {
 public:
 static shtns_cfg get(Index l_max, Index m_max, Index n_lon, Index n_lat);

 private:
  static std::array<Index, 4> current_config_;
  static shtns_cfg shtns_;
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
  using LatGrid = QuadratureLatitudeGrid<FejerQuadrature<double>, double>;
  using IndexVector = eigen::Vector<Index>;
  using ConstVectorMap = eigen::ConstVectorMap<double>;

  static SpectralCoeffs add_coeffs(const SHT &sht_l,
                                   SpectralCoeffsRef v,
                                   const SHT &sht_r,
                                   SpectralCoeffsRef w);

  static SpectralCoeffMatrix add_coeffs(const SHT &sht_inc_l,
                                        const SHT &sht_scat_l,
                                        SpectralCoeffMatrixRef v,
                                        const SHT &sht_inc_r,
                                        const SHT &sht_scat_r,
                                        SpectralCoeffMatrixRef w);

  static LatGrid get_latitude_grid(Index n_lat);
  static Vector get_longitude_grid(Index n_lon);

  /** Calculates the number of spherical harmonics coefficients for a real
   * transform.
   * @param l_max The maximum degree of the SHT.
   * @param m_max The maximum order of the SHT.
   * @return The number of spherical harmonics coefficients.
   */
  static Index calc_n_spectral_coeffs(Index l_max, Index m_max) {
    return (l_max + 1) * (m_max + 1) - (m_max * (m_max + 1)) / 2;
  }

  /** Calculates the number of spherical harmonics coefficients for a complex
   * transform.
   * @param l_max The maximum degree of the SHT.
   * @param m_max The maximum order of the SHT.
   * @return The number of spherical harmonics coefficients for a complex
   * transform.
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
    return static_cast<Index>(sqrt(2.0 * static_cast<double>(n_spectral_coeffs) + 0.25) - 1.5);
  }

  static std::array<Index, 4> get_params(Index n_lon, Index n_lat);

  /**
   * Create a spherical harmonics transformation object.
   *
   * @param l_max The maximum degree of the SHT.
   * @param m_max The maximum order of the SHT.
   * @param n_lon The number of longitude grid points.
   * @param n_lat The number of co-latitude grid points.
   */
  SHT(Index l_max, Index m_max, Index n_lon, Index n_lat);

  /**
   * Create a spherical harmonics transformation object.
   *
   * The values for n_lon and n_lat are set to 2 * l_max + 2 and
   * 2 * m_max + 2, respectively.
   *
   * @param l_max The maximum degree of the SHT.
   * @param m_max The maximum order of the SHT.
   */
  SHT(Index l_max, Index m_max);

  /**
   * Create a spherical harmonics transformation object.
   *
   * Create spherical harmonics transformation object with l_max == m_max
   * and values for n_lon and n_lat set to 2 * l_max + 2 and
   * 2 * m_max + 2, respectively.
   * @param l_max The maximum degree of the SHT.
   */
  SHT(Index l_max);

  /** Return latitude grid used by SHTns.
   * @return Eigen vector containing the latitude grid in radians.
   */
  LatGrid get_latitude_grid();

  /** Return co-latitude grid used by SHTns.
   * @return Eigen vector containing the co-latitude grid.
   */
  Vector get_colatitude_grid();

  Vector get_longitude_grid();

  /** L-indices of the SHT modes.
   *
   * @return A vector of indices containing the l-value corresponding to each
   * element in a spectral coefficient vector.
   */
  IndexVector get_l_indices();

  /** M-indices of the SHT modes.
   *
   * @return A vector of indices containing the m-value corresponding to each
   * element in a spectral coefficient vector.
   */
  IndexVector get_m_indices();

  /**
   * Copy spatial field into the array that holds spatial data for
   * spherical harmonics computations.
   *
   * @param m Eigen::Matrix or comparable providing read-only access
   * to the input data. Row indices should correspond to longitudes
   * (azimuth angle) and columns to latitudes (zenith angle).
   */
  void set_spatial_coeffs(const GridCoeffsRef &m) const;

  /**
   * Copy complex spatial field into the array that holds spatial data for
   * spherical harmonics computations.
   *
   * @param m Eigen::Matrix or comparable providing read-only access
   * to the input data. Row indices should correspond to longitudes
   * (azimuth angle) and columns to latitudes (zenith angle).
   */
  void set_spatial_coeffs(const CmplxGridCoeffsRef &m) const;

  /**
   * Copy spherical harmonics coefficients into the array that holds spectral
   * data for spherical harmonics computations.
   *
   * @param m SpectralCoeffs The spherical harmonics coefficients
   * representing the data.
   */
  void set_spectral_coeffs(const SpectralCoeffsRef &m) const;

  /**
   * Copy spherical harmonics coefficients into the array that holds spectral
   * data for spherical harmonics computations involving complex spatial fields.
   *
   * @param m Eigen vector containing the spherical harmonics coefficients
   * representing the data.
   */
  void set_spectral_coeffs_cmplx(const SpectralCoeffsRef &m) const;

  /**
   * Return content of the array that holds spatial data for
   * spherical harmonics computations.
   *
   * @return Eigen matrix containing the spatial field. Row indices should
   * correspond to longitudes (azimuth angle) and columns to latitudes (zenith
   * angle).
   */
  GridCoeffs get_spatial_coeffs() const;

  /**
   * Return content of the array that holds complex spatial data for
   * spherical harmonics computations.
   *
   * @return Eigen matrix containing the complex spatial field. Row indices
   * should correspond to longitudes (azimuth angle) and columns to latitudes
   * (zenith angle).
   */
  CmplxGridCoeffs get_cmplx_spatial_coeffs() const;

  /**
   * @return The size of the co-latitude grid.
   */
  Index get_n_latitudes() const { return n_lat_; }
  /**
   * @return The size of the longitude grid.
   */
  Index get_n_longitudes() const { return n_lon_; }
  /**
   * @return The number of spherical harmonics coefficients.
   */
  Index get_n_spectral_coeffs() const { return n_spectral_coeffs_; }
  /**
   * @return The number of spherical harmonics coefficients.
   */
  Index get_n_spectral_coeffs_cmplx() const { return n_spectral_coeffs_cmplx_; }

  /**
   * @return The maximum degree l of the SHT transformation.
   */
  Index get_l_max() { return l_max_; }

  /**
   * @return The maximum order m of the SHT transformation.
   */
  Index get_m_max() { return m_max_; }

  /**
   * Return content of the array that holds spectral data for
   * spherical harmonics computations.
   *
   * @return m Eigen vector containing the spherical harmonics coefficients
   * representing the data.
   */
  SpectralCoeffs get_spectral_coeffs() const;

  /**
   * Return content of the array that holds spectral data for
   * spherical harmonics computations of complex fields.
   *
   * @return m Eigen vector containing the spherical harmonics coefficients
   * representing the data.
   */
  SpectralCoeffs get_spectral_coeffs_cmplx() const;

  /** Apply forward SHT Transform
   *
   * Transforms discrete spherical data into spherical harmonics representation.
   * @param m GridCoeffs containing the data. Row indices should correspond to
   * longitudes (azimuth angle) and columns to latitudes (zenith angle).
   * @return Coefficient vector containing the spherical harmonics coefficients.
   */
  SpectralCoeffs transform(const GridCoeffsRef &m);

  /** Apply forward SHT Transform
   *
   * Transforms discrete spherical data into spherical harmonics representation.
   * @param m GridCoeffs containing the data. Row indices should correspond to
   * longitudes (azimuth angle) and columns to latitudes (zenith angle).
   * @return Coefficient vector containing the spherical harmonics coefficients.
   */
  SpectralCoeffs transform_cmplx(const CmplxGridCoeffsRef &m);

  /** Apply inverse SHT Transform
   *
   * Transforms discrete spherical data given in spherical harmonics
   * representation back to spatial domain.
   *
   * @param m SpectralCoeffs The spherical harmonics coefficients
   * representing the data.
   * @return GridCoeffs containing the spatial data.
   */
  GridCoeffs synthesize(const SpectralCoeffsRef &m);

  /** Apply inverse SHT Transform for complex data.
   *
   * Transforms discrete spherical data given in spherical harmonics
   * representation back to spatial domain.
   *
   * @param m SpectralCoeffs The spherical harmonics coefficients
   * representing the data.
   * @return GridCoeffs containing the spatial data.
   */
  CmplxGridCoeffs synthesize_cmplx(const SpectralCoeffsRef &m);

  /** Evaluate spectral representation at given point.
   *
   * @param m Spectral coefficient vector containing the SH coefficients.
   * @param phi The azimuth angles in radians.
   * @return theta The zenith angle in radians.
   */
  double evaluate(
      const SpectralCoeffsRef &m,
      double phi,
      double theta);

  /** Evaluate spectral representation at given point.
   *
   * @param m Spectral coefficient vector containing the SH coefficients.
   * @param points 2-row matrix containing the points (lon, lat) at which
   * to evaluate the function.
   * @return A vector containing the values corresponding to the points
   * in points.
   */
  eigen::Vector<double> evaluate(
      const SpectralCoeffsRef &m,
      const eigen::MatrixFixedRows<double, 2> &points);

  /** Evaluate 1D spectral representation at given point.
   *
   * This method covers the special case of 1D data that varies
   * only along latitudes. In this case the SH transform degenerates
   * to a Legendre transform.
   *
   * @param m Spectral coefficient vector containing the SH coefficients.
   * @param Vector containing the latitudes within [0, PI] to evaluate the
   * function.
   * @return A vector containing the values corresponding to the points
   * in points.
   */
  eigen::Vector<double> evaluate(const SpectralCoeffsRef &m,
                                 const eigen::Vector<double> &thetas);

 private:
  bool is_trivial_;
  Index l_max_, m_max_, n_lon_, n_lat_, n_spectral_coeffs_,
      n_spectral_coeffs_cmplx_;

  sht::FFTWArray<std::complex<double>> spectral_coeffs_, spectral_coeffs_cmplx_,
      cmplx_spatial_coeffs_;
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
   * @arg params Lenght-4 array containing the parameters required to initialize
   * the SHT transform: l_max, m_max, n_lat, n_lon. See documention of SHT class
   * for explanation of their significance.
   * @return Reference to SHT instance.
   */
  SHT &get_sht_instance(SHTParams params);

 protected:
  std::map<SHTParams, std::unique_ptr<SHT>> sht_instances_;
};

}  // namespace sht
}  // namespace scattering

#endif
