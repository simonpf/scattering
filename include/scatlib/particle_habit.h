/** \file particle_model.h
 *
 * Contains the ParticleHabit class which represents a model of a scattering particle,
 * which consists of a collection of scattering data for different particle sizes.
 *
 * @author Simon Pfreundschuh, 2020
 */
#ifndef __SCATLIB_PARTICLE_MODEL__
#define __SCATLIB_PARTICLE_MODEL__

#include <scatlib/single_scattering_data.h>
#include <scatlib/scattering_particle.h>

namespace scatlib {

// pxx :: export
/** Particle habit
 *
 * A particle habit represents a collection of scattering particles.
 */
class ParticleHabit {

public:

  /// Create an empty ParticleHabit.
  ParticleHabit() {};

  /// Create a ParticleHabit from given particles
ParticleHabit(std::vector<scatlib::ScatteringParticle> particles) : particles_(particles) {}

  /// Return vector contatining volume equivalent diameter of particles in the
  /// habit.
  eigen::Vector<double> get_d_eq() const {

    eigen::Vector<double> result(particles_.size());
    for (size_t i = 0; i < particles_.size(); ++i) {
      result[i] = particles_[i].get_d_eq();
    }
    return result;
  }

  /// Return vector contatining the maximum diameter of particles in the habit.
  eigen::Vector<double> get_d_max() const {
    eigen::Vector<double> result(particles_.size());
    for (size_t i = 0; i < particles_.size(); ++i) {
      result[i] = particles_[i].get_d_max();
    }
    return result;
  }

  /// Return vector contatining the mass of the particles in the habit.
  eigen::Vector<double> get_d_mass() const {
      eigen::Vector<double> result(particles_.size());
      for (size_t i = 0; i < particles_.size(); ++i) {
          result[i] = particles_[i].get_mass();
      }
      return result;
  }

  /** Scattering data describing habit.
   * @return A vector containing the single scattering data describing the particles in
   * in the habit.
   */
  const SingleScatteringData &get_single_scattering_data(size_t index) const {
    return particles_[index].get_data();
  }

  /** Interpolate single scattering data along frequencies.
   *
   * @param f_grid The frequency grid to which to interpolate the data.
   * @return A new particle habit with the data interpolated to the given
   * frequency grid.
   */
  ParticleHabit interpolate_frequency(eigen::Vector<double> f_grid) {
      auto f_grid_ptr = std::make_shared<eigen::Vector<double>>(f_grid);
      std::vector<scatlib::ScatteringParticle> new_data{};
      new_data.reserve(particles_.size());
      for (size_t i = 0; i < particles_.size(); ++i) {
          new_data.push_back(particles_[i].interpolate_frequency(f_grid_ptr));
      }
      return ParticleHabit(new_data);
  }

  /** Regrid the data to SHTns-compatible grids.
   *
   * In order to perform SHT transforms, the data must be provided on a regular
   * Fejer quadrature grid. This method will interpolate the data to the grids
   * required for correct SHT transformation. The grids are computed by linear
   * interpolation of the existing grids to the corresponding Fejer grids with
   * the same number of grid nodes.
   *
   * @return A new ParticleHabit object with the data interpolated to the angular
   * grids required for SHT transforms.
   */
    ParticleHabit regrid() const {
        std::vector<scatlib::ScatteringParticle> new_particles{};
        new_particles.reserve(particles_.size());
        for (size_t i = 0; i < particles_.size(); ++i) {
            new_particles[i] = particles_[i].regrid();
        }
        return ParticleHabit(new_particles);
    }

    /** Transform single scattering data in habit to spectral representation.
     *
     * 
     *
     */
    ParticleHabit to_spectral(Index l_max, Index m_max) {
        std::vector<scatlib::ScatteringParticle> new_particles{};
        new_particles.reserve(particles_.size());
        for (size_t i = 0; i < particles_.size(); ++i) {
            new_particles.push_back(particles_[i].to_spectral(l_max, m_max));
        }
        return ParticleHabit(new_particles);
    }

    ParticleHabit set_stokes_dim(Index n) const {
        std::vector<scatlib::ScatteringParticle> new_particles{};
        new_particles.reserve(particles_.size());
        for (size_t i = 0; i < particles_.size(); ++i) {
            new_particles.push_back(particles_[i].set_stokes_dim(n));
        }
        return ParticleHabit(new_particles);
    }


    ParticleHabit to_gridded(eigen::Vector<double> lon_inc,
                             eigen::Vector<double> lat_inc,
                             eigen::Vector<double> lon_scat,
                             eigen::Vector<double> lat_scat) {
      auto lon_inc_ptr = std::make_shared<eigen::Vector<double>>(lon_inc);
      auto lat_inc_ptr = std::make_shared<eigen::Vector<double>>(lat_inc);
      auto lon_scat_ptr = std::make_shared<eigen::Vector<double>>(lon_scat);
      auto lat_scat_ptr = std::make_shared<eigen::Vector<double>>(lat_scat);

      std::vector<scatlib::ScatteringParticle> new_particles{};
      new_particles.reserve(particles_.size());
      for (size_t i = 0; i < particles_.size(); ++i) {
          new_particles.push_back(particles_[i].to_gridded(lon_inc_ptr,
                                                      lat_inc_ptr,
                                                      lon_scat_ptr,
                                                           lat_scat_ptr));
      }
      return ParticleHabit(new_particles);
    }

    ParticleHabit to_lab_frame(eigen::Vector<double> lat_inc,
                               eigen::Vector<double> lon_scat,
                               eigen::Vector<double> lat_scat,
                               Index stokes_dim) {
        auto lat_inc_ptr = std::make_shared<eigen::Vector<double>>(lat_inc);
        auto lon_scat_ptr = std::make_shared<eigen::Vector<double>>(lon_scat);
        auto lat_scat_ptr = std::make_shared<eigen::Vector<double>>(lat_scat);
        std::vector<scatlib::ScatteringParticle> new_particles{};
        new_particles.reserve(particles_.size());
        for (size_t i = 0; i < particles_.size(); ++i) {
            new_particles.push_back(particles_[i].to_lab_frame(lat_inc_ptr, lon_scat_ptr, lat_scat_ptr, stokes_dim));
        }
        return ParticleHabit(new_particles);
    }

    ParticleHabit to_lab_frame(Index n_lat_inc, Index n_lon_scat, Index stokes_dim) const {
        std::vector<scatlib::ScatteringParticle> new_particles{};
        new_particles.reserve(particles_.size());
      for (size_t i = 0; i < particles_.size(); ++i) {
          new_particles.push_back(particles_[i].to_lab_frame(n_lat_inc, n_lon_scat, stokes_dim));
      }
      return ParticleHabit(new_particles);
    }

    ParticleHabit downsample_scattering_angles(const eigen::Vector<double> &lon_scat,
                                               const eigen::Vector<double> &lat_scat) const {
        std::vector<scatlib::ScatteringParticle> new_particles{};
        new_particles.reserve(particles_.size());
      auto lon_scat_ptr = std::make_shared<eigen::Vector<double>>(lon_scat);
      auto lat_scat_ptr = std::make_shared<eigen::Vector<double>>(lat_scat);
      for (size_t i = 0; i < particles_.size(); ++i) {
        new_particles[i] = particles_[i].downsample_scattering_angles(lon_scat_ptr,
                                                                      lat_scat_ptr);
      }
      return ParticleHabit(new_particles);
    }

    /** Calculate bulk scattering properties
     *
     * Calculates bulk scattering properties for a given temperature and particle number
     * distribution.
     *
     * @param The atmospheric temperature in K
     * @param pnd Vector containing the number of particles of each of the species in the model.
     * @return The scattering properties corresponding to the sum of the particle in the model
     * multiplied by the number given in pnd.
     */
    SingleScatteringData calculate_bulk_properties(
        double temperature,
        eigen::ConstVectorRef<double> pnd,
        double phase_function_norm) {
        assert(pnd.size() == particles_.size());

      auto temperature_vector = std::make_shared<eigen::Vector<double>>(1);
      (*temperature_vector)[0] = temperature;
      auto result = particles_[0].get_data().copy();
      if (result.get_t_grid().size() > 1) {
        result = result.interpolate_temperature(temperature_vector);
      }
      result *= pnd[0];

      for (Index i = 1; i < pnd.size(); ++i) {
        if (particles_[i].needs_t_interpolation()) {
          auto data = particles_[i].interpolate_temperature(temperature_vector);
          result += data.get_data() * pnd[i];
        } else {
          result += particles_[i].get_data() * pnd[i];
        }
      }
      result.normalize(phase_function_norm);
      return result;
    }

private:

    std::vector<scatlib::ScatteringParticle> particles_;
};

}

#endif
