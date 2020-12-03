#ifndef __SCATTERING_PARTICLE__
#define __SCATTERING_PARTICLE__

#include <scattering/single_scattering_data.h>

namespace scattering {

using eigen::Index;

struct ParticleProperties {
  std::string description = "";
  std::string source = "";
  std::string refractive_index = "";
  double mass = 0.0;
  double d_eq = 0.0;
  double d_max = 0.0;
  double d_aero = 0.0;
};

// pxx :: export
class Particle {
public:

Particle(ParticleProperties properties,
                   SingleScatteringData data)
      : data_(data), properties_(properties) {}

  Particle(double mass,
                     double d_eq,
                     double d_max,
                     SingleScatteringData data)
    : data_(data), properties_(ParticleProperties{"", "", "", mass, d_eq, d_max, 0.0}) {}

  Particle(const Particle &) = default;
  Particle &operator=(const Particle &) = default;

  double get_mass() const { return properties_.mass; }

  double get_d_max() const { return properties_.d_max; }

  double get_d_eq() const { return properties_.d_eq; }

  double get_d_aero() const { return properties_.d_aero; }

  const eigen::Vector<double>& get_f_grid() { return data_.get_f_grid(); }
  const eigen::Vector<double>& get_t_grid() { return data_.get_t_grid(); }
  const eigen::Vector<double>& get_lon_inc() { return data_.get_lon_inc(); }
  const eigen::Vector<double>& get_lat_inc() { return data_.get_lat_inc(); }
  const eigen::Vector<double>& get_lon_scat() { return data_.get_lon_scat(); }
  const eigen::Vector<double>& get_lat_scat() { return data_.get_lat_scat(); }

  //////////////////////////////////////////////////////////////////////////////
  // Manipulation of scattering data
  //////////////////////////////////////////////////////////////////////////////

  // pxx :: hide
  Particle interpolate_frequency(std::shared_ptr<eigen::Vector<double>> f_grid) const {
      return Particle(properties_, data_.interpolate_frequency(f_grid));
  }

  // pxx :: hide
  Particle interpolate_temperature(std::shared_ptr<eigen::Vector<double>> t_grid) {
      return Particle(properties_, data_.interpolate_temperature(t_grid));
  }

  SingleScatteringData interpolate_scattering_data(
      double temperature) const {
      auto t_grid = data_.get_t_grid();
      if (t_grid.size() == 1) {
          return data_;
      }

      auto l = t_grid[0];
      auto r = t_grid[1];
      auto lower_limit = l - 0.5 * (r - l);
      auto upper_limit = r + 0.5 * (r - l);

      auto temperature_vector = std::make_shared<eigen::Vector<double>>(1);
      (*temperature_vector)[0] = std::min(std::max(temperature, lower_limit), upper_limit);
      return data_.interpolate_temperature(temperature_vector);
  }

  // pxx :: hide
  Particle downsample_scattering_angles(std::shared_ptr<eigen::Vector<double>> lon_scat,
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
      auto data = data_.to_lab_frame(lat_inc_ptr, lon_scat_ptr, lat_scat_ptr, stokes_dim);
      return Particle(properties_, data);
  }

  Particle to_lab_frame(Index n_lat_inc,
                                  Index n_lon_scat,
                                  Index stokes_dim) const {
      auto data = data_.to_lab_frame(n_lat_inc, n_lon_scat, stokes_dim);
      return Particle(properties_, data);
  }

  Particle regrid() const {
      return Particle(properties_, data_.regrid());
  }

  Particle set_stokes_dim(Index n) const {
      auto data = data_.copy();
      data.set_stokes_dim(n);
      return Particle(properties_, data);
  }

  bool needs_t_interpolation() {
      return data_.get_t_grid().size() > 1;
  }


  const SingleScatteringData & get_data() const { return data_; }

 private:
  SingleScatteringData data_;
  ParticleProperties properties_;
};

}
#endif
