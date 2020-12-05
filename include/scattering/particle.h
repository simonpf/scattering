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
  Particle interpolate_frequency(
      std::shared_ptr<eigen::Vector<double>> f_grid) const {
    return Particle(properties_, data_.interpolate_frequency(f_grid));
  }

  // pxx :: hide
  Particle interpolate_temperature(
      std::shared_ptr<eigen::Vector<double>> t_grid) {
    return Particle(properties_, data_.interpolate_temperature(t_grid));
  }

  SingleScatteringData interpolate_scattering_data(double temperature) const {
    auto t_grid = data_.get_t_grid();
    auto n_temps = data_.get_n_temps();
    if (n_temps == 1) {
      return data_;
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
    return data_.interpolate_temperature(temperature_vector);
  }

  // pxx :: hide
  Particle downsample_scattering_angles(
      std::shared_ptr<eigen::Vector<double>> lon_scat,
      std::shared_ptr<eigen::Vector<double>> lat_scat) const {
    auto data = data_.downsample_scattering_angles(lon_scat, lat_scat);
    return Particle(properties_, data);
  }

  Particle to_spectral(Index l_max, Index m_max) const {
    return Particle(properties_, data_.to_spectral(l_max, m_max));
  }

  // pxx :: hide
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
  Particle to_lab_frame(std::shared_ptr<eigen::Vector<double>> lat_inc_ptr,
                        std::shared_ptr<eigen::Vector<double>> lon_scat_ptr,
                        std::shared_ptr<eigen::Vector<double>> lat_scat_ptr,
                        Index stokes_dim) const {
    auto data =
        data_.to_lab_frame(lat_inc_ptr, lon_scat_ptr, lat_scat_ptr, stokes_dim);
    return Particle(properties_, data);
  }

  Particle to_lab_frame(Index n_lat_inc,
                        Index n_lon_scat,
                        Index stokes_dim) const {
    auto data = data_.to_lab_frame(n_lat_inc, n_lon_scat, stokes_dim);
    return Particle(properties_, data);
  }

  Particle regrid() const { return Particle(properties_, data_.regrid()); }

  Particle set_stokes_dim(Index n) const {
    auto data = data_.copy();
    data.set_stokes_dim(n);
    return Particle(properties_, data);
  }

  bool needs_t_interpolation() { return data_.get_t_grid().size() > 1; }

  const SingleScatteringData& get_data() const { return data_; }

 private:
  SingleScatteringData data_;
  ParticleProperties properties_;
};

}  // namespace scattering
#endif
