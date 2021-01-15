/** \file particle.h
 *
 * Defines the Particle class which represents a particle through its
 * scattering, particle size and mass as well as additional meta information.
 *
 * @author Simon Pfreundschuh, 2020
 */
#ifndef __SCATTERING_PARTICLE__
#define __SCATTERING_PARTICLE__

#include <scattering/single_scattering_data.h>

namespace scattering {

using eigen::Index;

/** Properties of scattering particles.
 *
 * The ParticleProperties struct simply groups together meta data of a particle model
 * describing its size and mass as well as additional information regarding the
 * origin of the data.
 */
struct ParticleProperties {
  std::string name = "";
  std::string source = "";
  std::string refractive_index = "";
  double mass = 0.0;
  double d_eq = 0.0;
  double d_max = 0.0;
  double d_aero = 0.0;
};

// pxx :: export
/** A particle that scatters electro-magnetic radiation.
 *
 * The particle class holds the single scattering data describing the optical
 * properties of the particle together with other relevant properties of the
 * particle such as its size and mass.
 */
class Particle {
 public:
  Particle() {}

  /** Create particle from given properties and single-scattering data.
   *
   * @param ParticleProperties struct containing the particle meta data.
   * @param data SingleScatteringData object containing the single scattering
   *     data describing the particle.
   */
  Particle(ParticleProperties properties,
           SingleScatteringData data)
      : data_(data), properties_(properties) {}

  /** Create particle without minimum required meta information.
   *
   * This creates a particle with the minimum information required to use it is
   * a radiative transfer simulation.
   *
   * @param mass The mass of the particle in kg.
   * @param d_eq The volume-equivalent diameter of the particle in m.
   * @param d_max The maximum-diameter of the particle in m.
   * @param data SingleScatteringData object containing the single scattering
   *     data describing the particle.
   */
  Particle(double mass, double d_eq, double d_max, SingleScatteringData data)
      : data_(data),
        properties_(ParticleProperties{"", "", "", mass, d_eq, d_max, 0.0}) {}

  Particle(const Particle&) = default;
  Particle& operator=(const Particle&) = default;

  Particle copy() const { return Particle(properties_, data_.copy()); }

  //
  // Particle meta data.
  //

  /// The name of the particle, if available. Empty string otherwise.
  std::string get_name() const { return properties_.name; }
  /// The source of the particle data, if available. Empty string otherwise.
  std::string get_source() const { return properties_.source; }
  /// The refractive index of the particle data, if available. Empty string otherwise.
  std::string get_refractive_index() const { return properties_.refractive_index; }
  ParticleType get_particle_type() const { return data_.get_particle_type(); }
  DataFormat get_data_format() const { return data_.get_data_format(); }

  //
  // Particle size and mass.
  //

  /// The particle mass.
  double get_mass() const { return properties_.mass; }
  /// The particle maximum diameter.
  double get_d_max() const { return properties_.d_max; }
  /// The particle volume-equivalent diameter.
  double get_d_eq() const { return properties_.d_eq; }
  /// The aerodynamic cross-sectional area.
  double get_d_aero() const { return properties_.d_aero; }

  //
  // Scattering data.
  //

  /// The frequency grid over which the data is defined.
  const eigen::Vector<double>& get_f_grid() { return data_.get_f_grid(); }
  /// The temperature grid over which the data is defined.
  const eigen::Vector<double>& get_t_grid() { return data_.get_t_grid(); }
  /// Longitudinal component of the incoming angle.
  eigen::Vector<double> get_lon_inc() { return data_.get_lon_inc(); }
  /// Latitudinal component of the incoming angle.
  eigen::Vector<double> get_lat_inc() { return data_.get_lat_inc(); }
  /// Longitudinal component of the scattering (outgoing) angle.
  eigen::Vector<double> get_lon_scat() { return data_.get_lon_scat(); }
  /// Latitudinal component of the scattering (outgoing) angle.
  eigen::Vector<double> get_lat_scat() { return data_.get_lat_scat(); }

  //////////////////////////////////////////////////////////////////////////////
  // Manipulation of scattering data
  //////////////////////////////////////////////////////////////////////////////

  // pxx :: hide
  /** Frequency interpolation.
   *
   * Linear interpolation of scattering data along frequency dimension.
   *
   * @param f_grid The frequency grid to interpolate the data to.
   * @return New particle with the scattering data interpolated to the
   * given frequencies.
   */
  Particle interpolate_frequency(
      std::shared_ptr<eigen::Vector<double>> f_grid) const {
    return Particle(properties_, data_.interpolate_frequency(f_grid));
  }

  // pxx :: hide
  /** Temperature interpolation.
   *
   * Linear interpolation of scattering data along temperature dimension.
   *
   * @param t_grid The temperature grid to interpolate the data to.
   * @return New particle with the scattering data interpolated to the
   * given temperatures.
   */
  Particle interpolate_temperature(
      std::shared_ptr<eigen::Vector<double>> t_grid) {
    return Particle(properties_, data_.interpolate_temperature(t_grid));
  }

  /** Temperature interpolation with extrapolation.
   *
   * Linear interpolation of scattering data along temperature dimension.
   * Extrapolation is performed linearly up to the half-distance to the
   * second inner node. For values further out the limiting value from
   * the linear interpolation is used.
   *
   * @param temperature The temperature to interpolate the data to.
   * @return New particle with the scattering data interpolated to the
   * given temperature.
   */
  SingleScatteringData interpolate_temperature(double temperature) const {
    auto t_grid = data_.get_t_grid();
    auto n_temps = data_.get_n_temps();
    if (n_temps == 1) {
        return data_.copy();
    }

    auto l = t_grid[0];
    auto r = t_grid[1];
    auto lower_limit = l - 0.5 * (r - l);
    r = t_grid[n_temps - 1];
    l = t_grid[n_temps - 2];
    auto upper_limit = r + 0.5 * (r - l);

    auto temperature_vector = std::make_shared<eigen::Vector<double>>(1);
    (*temperature_vector)[0] =
        std::min(std::max(temperature, lower_limit), upper_limit);
    return data_.interpolate_temperature(temperature_vector, true);
  }

  // pxx :: hide
  /** Downsample scattering angles by averaging.
   *
   * This function downsamples the angular resolution of scattering data but keeps
   * the angular integral constant by averaging over the original data.
   *
   * @param lon_scat The scattering-angle longitude grid to downsample the data to.
   * @param lat_scat The scattering-angle latitude grid to downsample the data to.
   */
  Particle downsample_scattering_angles(
      std::shared_ptr<eigen::Vector<double>> lon_scat,
      std::shared_ptr<eigen::Vector<double>> lat_scat) const {
    auto data = data_.downsample_scattering_angles(lon_scat, lat_scat);
    return Particle(properties_, data);
  }

  /** Convert scattering data to spectral format.
   *
   * Converts scattering data to spectral format using a spherical
   * harmonics transform with the given parameters.
   *
   * @param l_max The l_max parameter to use for the SHT transformation.
   * @param m_max The m_max parameter to use for the SHT transformation.
   */
  Particle to_spectral(Index l_max, Index m_max) const {
    return Particle(properties_, data_.to_spectral(l_max, m_max));
  }

  // pxx :: hide
  /** Convert scattering data to gridded format.
   *
   * Converts scattering data to gridded format with given angular grids.
   *
   * @param lon_inc The incoming-angle longitude grid.
   * @param lat_inc The incoming-angle latitude grid.
   * @param lon_scat The scattering-angle longitude grid.
   * @param lat_scat The scattering-angle latitude grid.
   * @return A new scattering particle with the scattering data converted to gridded
   * format with the given angular grids.
   */
  Particle to_gridded(std::shared_ptr<eigen::Vector<double>> lon_inc,
                      std::shared_ptr<eigen::Vector<double>> lat_inc,
                      std::shared_ptr<eigen::Vector<double>> lon_scat,
                      std::shared_ptr<eigen::Vector<double>> lat_scat) const {
    auto gridded = data_.to_gridded().interpolate_angles(lon_inc,
                                                         lat_inc,
                                                         lon_scat,
                                                         lat_scat);
    return Particle(properties_, gridded);
  }

  // pxx :: hide
  /** Convert scattering data to lab frame.
   *
   * Converts scattering data of randomly-oriented particles from scattering frame
   * to lab frame.
   *
   * @param lat_inc_ptr The incoming-angle latitude grid for the output data.
   * @param lon_scat_ptr The scattering-angle longitude grid for the output data.
   * @param lat_scat_ptr The scattering-angle latitude grid for the output data.
   * @param stokes_dim The number of stokes dimensions to include in the output data.
   * @return A new scattering particle with the scattering data transformed to lab
   * frame on the given angular grids.
   */
  Particle to_lab_frame(std::shared_ptr<eigen::Vector<double>> lat_inc_ptr,
                        std::shared_ptr<eigen::Vector<double>> lon_scat_ptr,
                        std::shared_ptr<eigen::Vector<double>> lat_scat_ptr,
                        Index stokes_dim) const {
    auto data =
        data_.to_lab_frame(lat_inc_ptr, lon_scat_ptr, lat_scat_ptr, stokes_dim);
    return Particle(properties_, data);
  }

  /** Convert scattering data to lab frame.
   *
   * Converts scattering data of randomly-oriented particles from scattering frame
   * to lab frame.
   *
   * @param n_lat_inc The number grid points to use for the incoming and scattering
   * angle latitude grids.
   * @param n_lon_scat The number of grid points to use for the scattering angle
   * longitude grid.
   * @param stokes_dim The number of stokes dimensions to include in the output data.
   * @return A new scattering particle with the scattering data transformed to lab
   * frame on the given angular grids.
   */
  Particle to_lab_frame(Index n_lat_inc,
                        Index n_lon_scat,
                        Index stokes_dim) const {
    auto data = data_.to_lab_frame(n_lat_inc, n_lon_scat, stokes_dim);
    return Particle(properties_, data);
  }


  /** Regrid scattering data to SHT-conformant grids.
   *
   * Regrids the scattering data to grids that conform to the expectation
   * of the SHT transform. The grids are chose so that match the number of
   * current grid points.
   */
  Particle regrid() const { return Particle(properties_, data_.regrid()); }

  /** Reduce stokes dimension of data.
   * @param n The number of stokes dimensions to extract from the current data.
   * @return A new particle with the scattering data reduced to the given
   * stokes dimension.
   */
  Particle set_stokes_dim(Index n) const {
    auto data = data_.copy();
    data.set_stokes_dim(n);
    return Particle(properties_, data);
  }

  /// Whether data requires interpolation along t-grid.
  bool needs_t_interpolation() { return data_.get_t_grid().size() > 1; }

  //////////////////////////////////////////////////////////////////////////////
  // Extraction of scattering data.
  //////////////////////////////////////////////////////////////////////////////

  const SingleScatteringData& get_data() const { return data_; }

  /** Extract phase function.
   *
   * Converts scattering data to gridded format (if necessary) and returns
   * rank-6 tensor containing only the first component of the scattering data.
   *
   * @return Rank-6 tensor containing the first coefficient of gridded
   * phase-matrix data.
   */
  eigen::Tensor<double, 6> get_phase_function() const {
      return data_.get_phase_function();
  }
  /** Extract phase function.
   *
   * Converts scattering data to spectral format and return rank-5 tensor
   * containing only the first component of the scattering data.
   *
   * @return Rank-5 tensor containing the first coefficient of spectral
   * phase-matrix data.
   */
  eigen::Tensor<std::complex<double>, 5> get_phase_function_spectral() const {
      return data_.get_phase_function_spectral();
  }
  /** Extract phase matrix data.
   *
   * Converts scattering data to gridded format (if necessary) and returns
   * rank-7 tensor containing the full scattering data.
   *
   * @return Rank-7 tensor containing the gridded phase-matrix data.
   */
  eigen::Tensor<double, 7> get_phase_matrix_data() const {
      return data_.get_phase_matrix_data();
  }
  /** Extract phase matrix data.
   *
   * Converts scattering data to spectral format (if necessary) and returns
   * rank-6 tensor containing the full scattering data.
   *
   * @return Rank-6 tensor containing the spectral phase-matrix data.
   */
  eigen::Tensor<std::complex<double>, 6> get_phase_matrix_data_spectral() const {
      return data_.get_phase_matrix_data_spectral();
  }
  /** Extract scattering matrix data.
   *
   * Converts phase matrix data to gridded format (if necessary), assembles
   * scattering assembles scattering matrix data and returns rank-8 tensor
   * containing the scattering data in full scattering matrix. The last two
   * dimensions corresponding to the rows and columns of the scattering
   * matrix, respectively.
   *
   * @param stokes_dim The number of stokes dimensions of the scattering matrix
   * to return.
   * @return Rank-8 tensor containing the scattering matrix.
   */
  eigen::Tensor<double, 8> get_scattering_matrix(Index stokes_dim) const {
      return data_.get_scattering_matrix(stokes_dim);
  }

  //
  // Extinction matrix data.
  //

  /** Extract extinction coefficient.
   *
   * Converts extinction matrix data to gridded format (if necessary) and
   * returns rank-6 tensor containing the first coefficient of the
   * extinction matrix data.
   *
   * @return Rank-6 tensor containing the first component of the extinction matrix
   * data.
   */
  eigen::Tensor<double, 6> get_extinction_coeff() const {
      return data_.get_extinction_coeff();
  }

  /** Extract extinction matrix data.
   *
   * Converts extinction matrix data to gridded format (if necessary) and
   * returns rank-7 tensor containing the full extinction matrix data.
   *
   * @return Rank-7 tensor containing the extinction matrix
   * data.
   */
  eigen::Tensor<double, 7> get_extinction_matrix_data() const {
      return data_.get_extinction_matrix_data();
  }

  /** Extract extinction matrix.
   *
   * Converts extinction matrix data to gridded format (if necessary) and
   * returns rank-8 tensor containing extinction matrix data expanded to
   * full extinction matrices.
   *
   * The last two dimensions correspond to the rows and columns of the
   * scattering matrix, respectively.
   *
   *
   * @return Rank-7 tensor containing the extinction matrix data in compact
   * format.
   */
  eigen::Tensor<double, 8> get_extinction_matrix(Index stokes_dim) const {
    return data_.get_extinction_matrix(stokes_dim);
  }

  //
  // Absorption vector.
  //

  /** Extract absorption coefficient.
   *
   * Converts absorption vector data to gridded format (if necessary) and
   * returns rank-6 tensor containing the first coefficient of the
   * absorption vector data.
   *
   * @return Rank-6 tensor containing the first component of the absorption
   * vector data.
   */
  eigen::Tensor<double, 6> get_absorption_coeff() const {
      return data_.get_absorption_coeff();
  }

  /** Extract absorption vector data.
   *
   * Converts absorption vector data to gridded format (if necessary) and
   * returns rank-7 tensor containing the absorption vector in compact
   * format.
   *
   * @return Rank-7 tensor containing the absorption vector data.
   */
  eigen::Tensor<double, 7> get_absorption_vector_data() const {
      return data_.get_absorption_vector_data();
  }

  /** Extract and expand absorption vector.
   *
   * Converts absorption vector data to gridded format (if necessary) and
   * returns rank-7 tensor containing the absorption vector data expanded
   * to full stokes vector form.
   *
   * @param stokes_dim The stokes dimensions to which to expand the data.
   * @return Rank-7 tensor containing the absorption vector data.
   */
  eigen::Tensor<double, 7> get_absorption_vector(Index stokes_dim) const {
    return data_.get_absorption_vector(stokes_dim);
  }

  /** Forward scattering coefficient.
   *
   * Converts forward scattering coefficient to gridded format (if necessary) and
   * return the forward scattering coefficient data.
   *
   * @return Rank-7 tensor containing the forward scattering coefficient data.
   */
  eigen::Tensor<double, 7> get_forward_scattering_coeff() const {
    return data_.get_forward_scattering_coeff();
  }

  /** Backward scattering coefficient.
   *
   * Converts backward scattering coefficient to gridded format (if necessary) and
   * returns the forward scattering coefficient data.
   *
   * @return Rank-7 tensor containing the backward scattering coefficient data.
   */
  eigen::Tensor<double, 7> get_backward_scattering_coeff() const {
    return data_.get_backward_scattering_coeff();
  }

 private:
  SingleScatteringData data_;
  ParticleProperties properties_;
};

}  // namespace scattering
#endif
