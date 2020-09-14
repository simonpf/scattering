#ifndef __SCATLIB_ARTS_SSDB__
#define __SCATLIB_ARTS_SSDB__

#include "netcdfpp.hpp"
#include <regex>
#include <set>
#include <utility>
#include <filesystem>

#include <scatlib/eigen.h>
#include <scatlib/utils/array.h>
#include <scatlib/single_scattering_data.h>
#include <scatlib/particle_model.h>

namespace scatlib {

namespace arts_ssdb {

namespace detail {

/** Extract temperature and frequency from group name.
 * @group_name The name of one of the groups in an ASSDB particle file.
 * @return Pair(temp, freq) containing the extracted temperatures and
 *     frequency.
 */
std::pair<double, double> match_temp_and_freq(std::string group_name) {
  std::regex group_regex("Freq([0-9\.]*)GHz_T([0-9\.]*)K");
  std::smatch match;
  bool matches = std::regex_match(group_name, match, group_regex);
  if (matches) {
    double freq = std::stod(match[1]);
    double temp = std::stod(match[2]);
    return std::make_pair(freq, temp);
  }
  throw std::runtime_error("Group name doesn't match expected pattern.");
}

/** Extract particle metadata from filename.
 * @param file_name The filename as string.
 * @return tuple (match, d_eq, d_max, m) containing
 *    - match: Flag indicating whether the filename matches the ASSDB pattern.
 *    - d_eq: The volume-equivalent diameter
 *    - d_max: The maximum diameter.
 *    - m: The mass of the particle.
 */
std::tuple<bool, double, double, double> match_particle_properties(
    std::string file_name) {
  std::regex file_regex("Dmax([0-9]*)um_Dveq([0-9]*)um_Mass([-0-9\.e]*)kg\.nc");
  std::smatch match;
  bool matches = std::regex_match(file_name, match, file_regex);
  if (matches) {
    double d_max = std::stod(match[1]);
    double d_eq = std::stod(match[2]);
    double m = std::stod(match[3]);
    return std::make_tuple(true, d_eq, d_max, m);
  }
  return std::make_tuple(false, 0.0, 0.0, 0.0);
}

/** Indirect sort w.r.t. equivalent diameter.
 *
 * @param d_eq Vector containing the water equivalent diameter of the particles.
 * @param d_max Vector containing the maximum diameter of the particles.
 * @param m Vector containing the masses of the particle.
 */
void sort_by_d_eq(std::vector<double> &d_eq,
                  std::vector<double> &d_max,
                  std::vector<double> &m) {
    std::vector<size_t> indices{};
    indices.resize(d_eq.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;

    auto compare_d_eq = [&d_eq](size_t i, size_t j) {return d_eq[i] < d_eq[j];};
    std::sort(indices.begin(), indices.end(), compare_d_eq);

    std::vector<double> copy = d_eq;
    for (size_t i = 0; i < indices.size(); ++i) d_eq[i] = copy[indices[i]];
    copy = d_max;
    for (size_t i = 0; i < indices.size(); ++i) d_max[i] = copy[indices[i]];
    copy = m;
    for (size_t i = 0; i < indices.size(); ++i) m[i] = copy[indices[i]];
}

}

enum class Format {Gridded, Spectral, FullySpectral};

////////////////////////////////////////////////////////////////////////////////
// ScatteringData
////////////////////////////////////////////////////////////////////////////////

// pxx :: export
/** ASSD Scattering data.
 *
 * The scattering data for a given particle, temperature and frequency is
 * contained in a NetCDF group. This class provides an interface to these
 * groups and provides access to the scattering data.
 */
class ScatteringData {
  // pxx :: hide
  void determine_format() {
    format_ = Format::Gridded;
    if (group_.has_variable("phaMat_data_real")) {
      format_ = Format::Spectral;
    }
    if (group_.has_variable("extMat_data_real")) {
      format_ = Format::FullySpectral;
    }
  }

  // pxx :: hide
  template <typename Float>
  eigen::Vector<Float> get_vector(std::string name) {
    auto variable = group_.get_variable(name);
    auto size = variable.size();
    auto result = eigen::Vector<Float>{size};
    variable.read(result.data());
    return result;
  }

 public:
  ScatteringData(netcdf4::Group group) : group_(group) {
    temperature_ = group.get_variable("temperature").read<double>();
    frequency_ = group.get_variable("frequency").read<double>();
    determine_format();
  }

  double get_frequency() { return frequency_; }
  double get_temperature() { return temperature_; }

  ParticleType get_particle_type() {
    if (format_ == Format::Gridded) {
      auto phase_matrix_shape = group_.get_variable("phaMat_data").shape();
      if (phase_matrix_shape[0] == 6) {
        return ParticleType::Random;
      } else {
        return ParticleType::AzimuthallyRandom;
      }
    } else if (format_ == Format::Spectral) {
      auto phase_matrix_shape = group_.get_variable("phaMat_data_real").shape();
      if (phase_matrix_shape[0] == 6) {
        return ParticleType::Random;
      } else {
        return ParticleType::AzimuthallyRandom;
      }
    }
    return ParticleType::AzimuthallyRandom;
  }

  sht::SHT get_sht() {
      auto phase_matrix_dimensions = group_.get_variable("phaMat_data_real").shape();
      auto l_max = sht::SHT::calc_l_max(phase_matrix_dimensions[4]);
      return sht::SHT(l_max, l_max, l_max + 2 + l_max % 2, 2 * l_max);
  }

  eigen::Vector<double> get_f_grid() {return eigen::Vector<double>::Constant(frequency_, 1);}
  eigen::Vector<double> get_t_grid() {return eigen::Vector<double>::Constant(temperature_, 1);}
  eigen::Vector<double> get_lon_inc() { return get_vector<double>("aa_inc"); }
  eigen::Vector<float> get_lon_inc_spectral() { return get_vector<float>("aa_inc"); }
  eigen::Vector<double> get_lat_inc() { return get_vector<double>("za_inc"); }
  eigen::Vector<float> get_lat_inc_spectral() { return get_vector<float>("za_inc"); }
  eigen::Vector<double> get_lon_scat() { return get_vector<double>("aa_scat"); }
  eigen::Vector<double> get_lat_scat() { return get_vector<double>("za_scat"); }

  eigen::Tensor<double, 7> get_phase_matrix_data_gridded() {
    // Load data from file.
    auto variable = group_.get_variable("phaMat_data");
    auto dimensions = variable.get_shape_array<eigen::Index, 5>();
    eigen::Tensor<double, 5> result{dimensions};
    variable.read(result.data());

    // Reshape and shuffle data.
    auto shuffle_dimensions = make_array<eigen::Index>(1, 2, 3, 4, 0);
    auto result_shuffled = result.shuffle(shuffle_dimensions);
    auto new_dimensions = concat(make_array<eigen::Index>(1, 1), dimensions);
    auto result_reshaped = result_shuffled.reshape(new_dimensions);
    return result_reshaped;
  }

  eigen::Tensor<std::complex<double>, 6> get_phase_matrix_data_spectral() {
    // Read data from file.
    auto variable_real = group_.get_variable("phaMat_data_real");
    auto variable_imag = group_.get_variable("phaMat_data_imag");
    auto dimensions = variable_real.get_shape_array<eigen::Index, 4>();
    eigen::Tensor<float, 4> real{dimensions};
    eigen::Tensor<float, 4> imag{dimensions};
    variable_real.read(real.data());
    variable_imag.read(imag.data());
    eigen::Tensor<std::complex<double>, 4> result =
        imag.cast<std::complex<double>>();
    result = result * std::complex<double>(0.0, 1.0);
    result += real;

    // Reshape and shuffle data.
    auto shuffle_dimensions = make_array<eigen::Index>(1, 2, 3, 0);
    auto result_shuffled = result.shuffle(shuffle_dimensions);
    auto new_dimensions = concat(make_array<eigen::Index>(1, 1), dimensions);
    auto result_reshaped = result_shuffled.reshape(new_dimensions);
    return result_reshaped;
  }

  eigen::Tensor<double, 7> get_extinction_matrix_data_gridded() {
    // Read data from file.
    auto variable = group_.get_variable("extMat_data");
    auto dimensions = variable.get_shape_array<eigen::Index, 3>();
    eigen::Tensor<double, 3> result{dimensions};
    variable.read(result.data());

    // Reshape and shuffle data.
    auto shuffle_dimensions = make_array<eigen::Index>(1, 2, 0);
    auto result_shuffled = result.shuffle(shuffle_dimensions);
    auto new_dimensions = concat(
        make_array<eigen::Index>(1, 1),
        concat(take<0, 1>(dimensions),
               concat(make_array<eigen::Index>(1, 1), take<2>(dimensions))));
    auto result_reshaped = result_shuffled.reshape(new_dimensions);
    return result_reshaped;
  }

  eigen::Tensor<std::complex<double>, 6> get_extinction_matrix_data_spectral() {
    // Read data from file.
    auto variable = group_.get_variable("extMat_data");
    auto dimensions = variable.get_shape_array<eigen::Index, 3>();
    eigen::Tensor<float, 3> result{dimensions};
    variable.read(result.data());

    // Reshape and shuffle data.
    auto shuffle_dimensions = make_array<eigen::Index>(1, 2, 0);
    auto result_shuffled = result.shuffle(shuffle_dimensions);
    auto new_dimensions = concat(
        make_array<eigen::Index>(1, 1),
        concat(take<0, 1>(dimensions),
               concat(make_array<eigen::Index>(1), take<2>(dimensions))));
    eigen::Tensor<float, 6> result_reshaped = result_shuffled.reshape(new_dimensions);
    return result_reshaped.cast<std::complex<double>>();
  }

  eigen::Tensor<double, 7> get_absorption_vector_data_gridded() {
    auto variable = group_.get_variable("absVec_data");
    auto dimensions = variable.get_shape_array<eigen::Index, 3>();
    eigen::Tensor<double, 3> result{dimensions};
    variable.read(result.data());

    // Reshape and shuffle data.
    auto shuffle_dimensions = make_array<eigen::Index>(1, 2, 0);
    auto result_shuffled = result.shuffle(shuffle_dimensions);
    auto new_dimensions = concat(
        make_array<eigen::Index>(1, 1),
        concat(take<0, 1>(dimensions),
               concat(make_array<eigen::Index>(1, 1), take<2>(dimensions))));
    auto result_reshaped = result_shuffled.reshape(new_dimensions);
    return result_reshaped;
  }

  eigen::Tensor<std::complex<double>, 6> get_absorption_vector_data_spectral() {
    auto variable = group_.get_variable("absVec_data");
    auto dimensions = variable.get_shape_array<eigen::Index, 3>();
    eigen::Tensor<float, 3> result{dimensions};
    variable.read(result.data());

    // Reshape and shuffle data.
    auto shuffle_dimensions = make_array<eigen::Index>(1, 2, 0);
    auto result_shuffled = result.shuffle(shuffle_dimensions);
    auto new_dimensions = concat(
        make_array<eigen::Index>(1, 1),
        concat(take<0, 1>(dimensions),
               concat(make_array<eigen::Index>(1), take<2>(dimensions))));
    eigen::Tensor<float, 6> result_reshaped = result_shuffled.reshape(new_dimensions);
    return result_reshaped.cast<std::complex<double>>();
  }

  eigen::Tensor<double, 7> get_backward_scattering_coeff_data_gridded() {
      auto phase_matrix = get_phase_matrix_data_gridded();
      auto dimensions = phase_matrix.dimensions();
      auto backward_scattering_coeff = phase_matrix.chip<5>(dimensions[5] - 1);
      dimensions[5] = 1;
      return backward_scattering_coeff.reshape(dimensions);
  }

  eigen::Tensor<std::complex<double>, 6> get_backward_scattering_coeff_data_spectral() {
      auto phase_matrix = get_phase_matrix_data_spectral();
      auto sht = get_sht();

      auto data_spectral = ScatteringDataFieldSpectral(get_f_grid(),
                                                       get_t_grid(),
                                                       get_lon_inc(),
                                                       get_lat_inc(),
                                                       sht,
                                                       get_phase_matrix_data_spectral());
      auto data_gridded = data_spectral.to_gridded();
      auto phase_matrix_gridded = data_gridded.get_data();
      auto dimensions = phase_matrix_gridded.dimensions();
      auto forward_scattering_coeff = phase_matrix.chip<4>(0).chip<4>(dimensions[5] - 1);
      auto dimensions_output = phase_matrix.dimensions();
      dimensions_output[4] = 1;
      return forward_scattering_coeff.cast<std::complex<double>>().reshape(dimensions_output);
  }

  eigen::Tensor<double, 7> get_forward_scattering_coeff_data_gridded() {
      auto phase_matrix = get_phase_matrix_data_gridded();
      auto dimensions = phase_matrix.dimensions();
      auto backward_scattering_coeff = phase_matrix.chip<5>(0);
      dimensions[5] = 1;
      return backward_scattering_coeff.reshape(dimensions);
  }

  eigen::Tensor<std::complex<double>, 6> get_forward_scattering_coeff_data_spectral() {
      auto phase_matrix = get_phase_matrix_data_spectral();
      auto sht = get_sht();

      auto data_spectral = ScatteringDataFieldSpectral(get_f_grid(),
                                                       get_t_grid(),
                                                       get_lon_inc(),
                                                       get_lat_inc(),
                                                       sht,
                                                       get_phase_matrix_data_spectral());
      auto data_gridded = data_spectral.to_gridded();
      auto phase_matrix_gridded = data_gridded.get_data();
      auto forward_scattering_coeff = phase_matrix.chip<4>(0).chip<4>(0);
      auto dimensions_output = phase_matrix.dimensions();
      dimensions_output[4] = 1;
      return forward_scattering_coeff.cast<std::complex<double>>().reshape(dimensions_output);
  }

  operator SingleScatteringDataGridded<double>() {
    assert(format_ = Format::Gridded);

    auto dummy_grid = std::make_shared<eigen::Vector<double>>(1);

    auto f_grid = std::make_shared<eigen::Vector<double>>(get_f_grid());
    auto t_grid = std::make_shared<eigen::Vector<double>>(get_t_grid());
    auto lon_inc = std::make_shared<eigen::Vector<double>>(get_lon_inc());
    auto lat_inc = std::make_shared<eigen::Vector<double>>(get_lat_inc());
    auto lon_scat = std::make_shared<eigen::Vector<double>>(get_lon_scat());
    auto lat_scat = std::make_shared<eigen::Vector<double>>(get_lat_scat());
    auto phase_matrix =
        std::make_shared<eigen::Tensor<double, 7>>(get_phase_matrix_data_gridded());
    auto extinction_matrix = std::make_shared<eigen::Tensor<double, 7>>(
        get_extinction_matrix_data_gridded());
    auto absorption_vector = std::make_shared<eigen::Tensor<double, 7>>(
        get_absorption_vector_data_gridded());
    auto backward_scattering_coeff = std::make_shared<eigen::Tensor<double, 7>>(
        get_backward_scattering_coeff_data_gridded());
    auto forward_scattering_coeff = std::make_shared<eigen::Tensor<double, 7>>(
        get_backward_scattering_coeff_data_gridded());
    return SingleScatteringDataGridded<double>(f_grid,
                                               t_grid,
                                               lon_inc,
                                               lat_inc,
                                               lon_scat,
                                               lat_scat,
                                               phase_matrix,
                                               extinction_matrix,
                                               absorption_vector,
                                               backward_scattering_coeff,
                                               forward_scattering_coeff);
  }

  operator SingleScatteringDataSpectral<double>() {
      assert(format_ = Format::Spectral);

      auto dummy_grid = std::make_shared<eigen::Vector<double>>(1);

    auto f_grid = std::make_shared<eigen::Vector<double>>(get_f_grid());
    auto t_grid = std::make_shared<eigen::Vector<double>>(get_t_grid());
    auto lon_inc = std::make_shared<eigen::Vector<double>>(get_lon_inc());
    auto lat_inc = std::make_shared<eigen::Vector<double>>(get_lat_inc());
    auto sht = std::make_shared<sht::SHT>(get_sht());
    auto phase_matrix =
        std::make_shared<eigen::Tensor<std::complex<double>, 6>>(get_phase_matrix_data_spectral());
    auto extinction_matrix = std::make_shared<eigen::Tensor<std::complex<double>, 6>>(
        get_extinction_matrix_data_spectral());
    auto absorption_vector = std::make_shared<eigen::Tensor<std::complex<double>, 6>>(
        get_absorption_vector_data_spectral());
    auto backward_scattering_coeff = std::make_shared<eigen::Tensor<std::complex<double>, 6>>(
        get_backward_scattering_coeff_data_spectral());
    auto forward_scattering_coeff = std::make_shared<eigen::Tensor<std::complex<double>, 6>>(
        get_backward_scattering_coeff_data_spectral());
    return SingleScatteringDataSpectral<double>(f_grid,
                                                t_grid,
                                                lon_inc,
                                                lat_inc,
                                                sht,
                                                phase_matrix,
                                                extinction_matrix,
                                                absorption_vector,
                                                backward_scattering_coeff,
                                                forward_scattering_coeff);

  }

  operator SingleScatteringData() {
    SingleScatteringDataImpl* data = nullptr;
    if (format_ == Format::Gridded) {
      data = new SingleScatteringDataGridded<double>(*this);
    } else if (format_ == Format::Spectral) {
      data = new SingleScatteringDataSpectral<double>(*this);
    }
    return SingleScatteringData(data);
  }

 private:
  Format format_;
  double temperature_, frequency_;
  netcdf4::Group group_;
};

////////////////////////////////////////////////////////////////////////////////
// ParticleFile
////////////////////////////////////////////////////////////////////////////////

// pxx :: export
/** ASSD Scattering data.
 *
 * The scattering data for a particle of given size is contained in a single
 * NetCDF file. This class provides an interface to these files and access
 * to the groups, which contain the scattering data for given temperatures
 * and frequencies.
 */
// pxx :: export
class ParticleFile {
  // pxx :: hide
  void parse_temps_and_freqs() {
    auto group_names = file_handle_.get_group_names();
    std::set<double> freqs;
    std::set<double> temps;
    for (auto &name : group_names) {
      auto freq_and_temp = detail::match_temp_and_freq(name);
      freqs.insert(std::get<0>(freq_and_temp));
      temps.insert(std::get<1>(freq_and_temp));
      group_map_[freq_and_temp] = file_handle_.get_group(name);
    }
    freqs_.resize(freqs.size());
    std::copy(freqs.begin(), freqs.end(), freqs_.begin());
    temps_.resize(temps.size());
    std::copy(temps.begin(), temps.end(), temps_.begin());
    std::sort(freqs_.begin(), freqs_.end());
    std::sort(temps_.begin(), temps_.end());
  }


 public:

  class DataIterator;

  ParticleFile(std::string path) : file_handle_(netcdf4::File::open(path)) {
      auto properties = detail::match_particle_properties(path);
      d_eq_ = std::get<1>(properties);
      d_max_ = std::get<2>(properties);
      mass_ = std::get<3>(properties);
      parse_temps_and_freqs();
  }

  ParticleType get_particle_type() {
    auto f = freqs_[0];
    auto t = temps_[0];
    return ScatteringData(group_map_[std::make_pair(f, t)]).get_particle_type();
  }

  double get_d_eq() {return d_eq_;}
  double get_d_max() {return d_max_;}
  double get_mass() {return mass_;}

  std::vector<double> get_frequencies() {
      return freqs_;
  }

  eigen::Vector<double> get_f_grid() {
      return eigen::VectorMap<double>(freqs_.data(), freqs_.size());
  }

  std::vector<double> get_temperatures() {
      return temps_;
  }

  eigen::Vector<double> get_t_grid() {
      return eigen::VectorMap<double>(temps_.data(), temps_.size());
  }

  DataIterator begin();
  DataIterator end();

  ScatteringData get_scattering_data(size_t i, size_t j) {
      double freq = freqs_[i];
      double temp = temps_[j];
      auto found = group_map_.find(std::make_pair(freq, temp));
      auto group = found->second.get_group("SingleScatteringData");
      return ScatteringData(group);
  }

  operator SingleScatteringData() {
      auto f_grid = get_f_grid();
      auto t_grid = get_t_grid();

      auto first = ScatteringData(group_map_[std::make_pair(f_grid[0], t_grid[0])]);
      auto lon_inc = first.get_lon_inc();
      auto lat_inc = first.get_lat_inc();
      auto lon_scat = first.get_lon_scat();
      auto lat_scat = first.get_lat_scat();

      auto result = SingleScatteringData(f_grid,
                                         t_grid,
                                         lon_inc,
                                         lat_inc,
                                         lon_scat,
                                         lat_scat,
                                         first.get_particle_type());
      for (size_t i = 0; i < freqs_.size(); ++i) {
          for (size_t j = 0; j < temps_.size(); ++j) {
              auto data = get_scattering_data(i, j);
              result.set_data(i, j, data);
          }
      }
      return result;
  }

 private:
  double d_eq_, d_max_, mass_;
  std::vector<double> freqs_;
  std::vector<double> temps_;
  std::map<std::pair<double, double>, netcdf4::Group> group_map_;
  netcdf4::File file_handle_;
};

class ParticleFile::DataIterator {
public:
DataIterator(const ParticleFile *file,
             size_t f_index = 0,
             size_t t_index = 0)
    : file_(file),
      f_index_(f_index),
      t_index_(t_index) {}

    DataIterator& operator++() {
        t_index_++;
        if (t_index_ >= file_->temps_.size()) {
            f_index_ ++;
            t_index_ = 0;
        }
        return *this;
    }

    bool operator==(const DataIterator &other) const {return (t_index_ == other.t_index_) && (f_index_ == other.f_index_);}
    bool operator!=(const DataIterator &other) const {return !(*this == other);}
    ScatteringData operator*() {
        auto f = file_->freqs_[f_index_];
        auto t = file_->temps_[t_index_];
        return file_->group_map_.find(std::make_pair(f, t))->second;
    }

    double get_frequency() {
        return file_->freqs_[f_index_];
    }

    double get_temperature() {
        return file_->temps_[t_index_];
    }

    // iterator traits
    using difference_type = size_t;
    using value_type = ScatteringData;
    using pointer = const ScatteringData*;
    using reference = const ScatteringData&;
    using iterator_category = std::forward_iterator_tag;

private:
    const ParticleFile *file_ = nullptr;
    size_t f_index_, t_index_;
};

ParticleFile::DataIterator ParticleFile::begin() {
    return DataIterator(this, 0, 0);
}

ParticleFile::DataIterator ParticleFile::end() {
    return DataIterator(this, freqs_.size(), 0);
}

////////////////////////////////////////////////////////////////////////////////
// Habit Folder
////////////////////////////////////////////////////////////////////////////////

// pxx :: export
/** A folder describing a particle habit.
 *
 * Habits in the database are represented by a folder containing NetCDF4 files
 * for each available particle size. This class parses such a folder and provides
 * access to each particle of the habit.
 */
class HabitFolder {

  // pxx :: hide
  void parse_files() {
    std::vector<double> d_eq_vec, d_max_vec, mass_vec;
    auto it = std::filesystem::directory_iterator(base_path_);
    for (auto &p : it) {
      auto match = detail::match_particle_properties(p.path().filename());
      if (std::get<0>(match)) {
        double d_eq = std::get<1>(match);
        double d_max = std::get<2>(match);
        double mass = std::get<3>(match);
        d_eq_vec.push_back(d_eq);
        d_max_vec.push_back(d_max);
        mass_vec.push_back(mass);
        files_[d_eq] = base_path_ / p.path();
      }
    }
    detail::sort_by_d_eq(d_eq_vec, d_max_vec, mass_vec);
    d_eq_ = eigen::VectorMap<double>(d_eq_vec.data(), d_eq_vec.size());
    d_max_ = eigen::VectorMap<double>(d_max_vec.data(), d_max_vec.size());
    mass_ = eigen::VectorMap<double>(mass_vec.data(), mass_vec.size());
  }

public:

  class DataIterator;

HabitFolder(std::string path) : base_path_(path) {
      parse_files();
 }

  eigen::Vector<double> get_d_eq() {
      return d_eq_;
  }

  eigen::Vector<double> get_d_max() {
      return d_max_;
  }

  eigen::Vector<double> get_mass() {
      return mass_;
  }

  DataIterator begin();
  DataIterator end();

  operator ParticleModel() {
    std::vector<SingleScatteringData> data;
    data.reserve(files_.size());
    for (auto &d : d_eq_) {
      data.push_back(ParticleFile(files_[d]));
    }
    return ParticleModel(get_d_eq(), get_d_max(), get_mass(), data);
  }

  ParticleModel to_particle_model() { return *this; }

 private:
  std::filesystem::path base_path_;
  eigen::Vector<double> d_eq_;
  eigen::Vector<double> d_max_;
  eigen::Vector<double> mass_;
    std::map<double, std::filesystem::path> files_;
};

class HabitFolder::DataIterator {
public:
DataIterator(const HabitFolder *folder, size_t index = 0)
    : folder_(folder),
      index_(index) {}

    DataIterator& operator++() {index_++; return *this;}
    bool operator==(const DataIterator &other) const {return (index_ == other.index_);}
    bool operator!=(const DataIterator &other) const {return !(*this == other);}
    ParticleFile operator*() {
        double d_eq = folder_->d_eq_[index_];
        return ParticleFile(folder_->files_.find(d_eq)->second);
    }

    double get_d_eq() {return folder_->d_eq_[index_];}
    double get_d_max() {return folder_->d_max_[index_];}
    double get_mass() {return folder_->mass_[index_];}

    // iterator traits
    using difference_type = size_t;
    using value_type = ScatteringData;
    using pointer = const ScatteringData*;
    using reference = const ScatteringData&;
    using iterator_category = std::forward_iterator_tag;

private:
    const HabitFolder *folder_ = nullptr;
    size_t index_;
};

HabitFolder::DataIterator HabitFolder::begin() {
    return DataIterator(this, 0);
}

HabitFolder::DataIterator HabitFolder::end() {
    return DataIterator(this, d_eq_.size());
}

}  // namespace arts_ssdb
}
#endif
