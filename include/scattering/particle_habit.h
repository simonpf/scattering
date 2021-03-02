/** \file particle_habit.h
 *
 * Contains the ParticleHabit class which represents a habit of a scattering particle,
 * which consists of a collection of scattering data for different particle sizes.
 *
 * @author Simon Pfreundschuh, 2020
 */
#ifndef __SCATTERING_PARTICLE_HABIT__
#define __SCATTERING_PARTICLE_HABIT__

#include <scattering/single_scattering_data.h>
#include <scattering/particle.h>

namespace scattering {

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
  ParticleHabit(const std::vector<scattering::Particle> &particles)
      : particles_(particles) {}

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
  eigen::Vector<double> get_mass() const {
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
  ParticleHabit interpolate_frequency(eigen::VectorPtr<double> f_grid) {
      std::vector<scattering::Particle> new_data{};
      new_data.reserve(particles_.size());
      for (size_t i = 0; i < particles_.size(); ++i) {
          new_data.push_back(particles_[i].interpolate_frequency(f_grid));
      }
      return ParticleHabit(new_data);
  }

  ParticleHabit interpolate_frequency(eigen::Vector<double> f_grid) {
      auto f_grid_ptr = std::make_shared<eigen::Vector<double>>(f_grid);
      return interpolate_frequency(f_grid_ptr);
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
        std::vector<scattering::Particle> new_particles{};
        new_particles.reserve(particles_.size());
        for (size_t i = 0; i < particles_.size(); ++i) {
            new_particles.push_back(particles_[i].regrid());
        }
        return ParticleHabit(new_particles);
    }

    /** Transform single scattering data in habit to spectral representation.
     *
     *
     */
    ParticleHabit to_spectral(Index l_max, Index m_max) {
        std::vector<scattering::Particle> new_particles{};
        new_particles.reserve(particles_.size());
        for (size_t i = 0; i < particles_.size(); ++i) {
            new_particles.push_back(particles_[i].to_spectral(l_max, m_max));
        }
        return ParticleHabit(new_particles);
    }

    ParticleHabit set_stokes_dim(Index n) const {
        std::vector<scattering::Particle> new_particles{};
        new_particles.reserve(particles_.size());
        for (size_t i = 0; i < particles_.size(); ++i) {
            new_particles.push_back(particles_[i].set_stokes_dim(n));
        }
        return ParticleHabit(new_particles);
    }


    ParticleHabit to_gridded(eigen::VectorPtr<double> lon_inc,
                             eigen::VectorPtr<double> lat_inc,
                             eigen::VectorPtr<double> lon_scat,
                             LatitudeGridPtr<double> lat_scat) {
      std::vector<scattering::Particle> new_particles{};
      new_particles.reserve(particles_.size());
      for (size_t i = 0; i < particles_.size(); ++i) {
          new_particles.push_back(particles_[i].to_gridded(lon_inc,
                                                           lat_inc,
                                                           lon_scat,
                                                           lat_scat));
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
      auto lat_scat_ptr = std::make_shared<IrregularLatitudeGrid<double>>(lat_scat);
      return to_gridded(lon_inc_ptr, lat_inc_ptr, lon_scat_ptr, lat_scat_ptr);
    }

    ParticleHabit to_lab_frame(eigen::VectorPtr<double> lat_inc,
                               eigen::VectorPtr<double> lon_scat,
                               LatitudeGridPtr<double> lat_scat,
                               Index stokes_dim) {
        std::vector<scattering::Particle> new_particles{};
        new_particles.reserve(particles_.size());
        for (size_t i = 0; i < particles_.size(); ++i) {
            new_particles.push_back(particles_[i].to_lab_frame(lat_inc, lon_scat, lat_scat, stokes_dim));
        }
        return ParticleHabit(new_particles);
    }

    ParticleHabit to_lab_frame(eigen::Vector<double> lat_inc,
                               eigen::Vector<double> lon_scat,
                               eigen::Vector<double> lat_scat,
                               Index stokes_dim) {
        auto lat_inc_ptr = std::make_shared<eigen::Vector<double>>(lat_inc);
        auto lon_scat_ptr = std::make_shared<eigen::Vector<double>>(lon_scat);
        auto lat_scat_ptr = std::make_shared<IrregularLatitudeGrid<double>>(lat_scat);
        return to_lab_frame(lat_inc_ptr, lon_scat_ptr, lat_scat_ptr, stokes_dim);
    }

    ParticleHabit to_lab_frame(Index n_lat_inc, Index n_lon_scat, Index stokes_dim) const {
        std::vector<scattering::Particle> new_particles{};
        new_particles.reserve(particles_.size());
      for (size_t i = 0; i < particles_.size(); ++i) {
          new_particles.push_back(particles_[i].to_lab_frame(n_lat_inc, n_lon_scat, stokes_dim));
      }
      return ParticleHabit(new_particles);
    }

    ParticleHabit downsample_scattering_angles(eigen::VectorPtr<double> lon_scat,
                                               std::shared_ptr<LatitudeGrid<double>> lat_scat) const {
        std::vector<scattering::Particle> new_particles{};
        new_particles.reserve(particles_.size());
        for (size_t i = 0; i < particles_.size(); ++i) {
            new_particles.push_back(particles_[i].downsample_scattering_angles(lon_scat,
                                                                               lat_scat));
        }
        return ParticleHabit(new_particles);
    }

    ParticleHabit downsample_scattering_angles(const eigen::Vector<double> &lon_scat,
                                               const eigen::Vector<double> &lat_scat) const {
      auto lon_scat_ptr = std::make_shared<eigen::Vector<double>>(lon_scat);
      auto lat_scat_ptr = std::make_shared<IrregularLatitudeGrid<double>>(lat_scat);
      return downsample_scattering_angles(lon_scat_ptr, lat_scat_ptr);
    }

    /** Calculate bulk scattering properties
     *
     * Calculates bulk scattering properties for a given temperature and particle number
     * distribution.
     *
     * @param The atmospheric temperature in K
     * @param pnd Vector containing the number of particles of each of the species in the habit.
     * @return The scattering properties corresponding to the sum of the particle in the habit
     * multiplied by the number given in pnd.
     */
    SingleScatteringData calculate_bulk_properties(
        double temperature,
        eigen::ConstVectorRef<double> pnd) {
        assert(static_cast<size_t>(pnd.size()) == particles_.size());

      auto temperature_vector = std::make_shared<eigen::Vector<double>>(1);
      (*temperature_vector)[0] = temperature;
      auto result = particles_[0].interpolate_temperature(temperature);
      result *= pnd[0];

      for (Index i = 1; i < pnd.size(); ++i) {
        auto data = particles_[i].interpolate_temperature(temperature);
        result += data * pnd[i];
      }
      result.normalize(1.0);
      return result;
    }

private:

    std::vector<scattering::Particle> particles_;
};

}

#endif
