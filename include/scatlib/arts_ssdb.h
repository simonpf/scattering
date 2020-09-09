#ifndef __SCATLIB_ARTS_SSDB__
#define __SCATLIB_ARTS_SSDB__

#include "netcdfpp.hpp"
#include <regex>
#include <set>
#include <utility>
#include <filesystem>

#include <scatlib/eigen.h>

namespace scatlib {

namespace arts_ssdb {

namespace detail {

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

std::tuple<bool, double, double, double> match_particle_file_name(
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

// pxx :: export
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

 public:
  ScatteringData(netcdf4::Group group) : group_(group) {}

  eigen::Tensor<double, 5> get_phase_matrix_data_gridded() {
    auto variable = group_.get_variable("phaMat_data");
    auto dimensions = variable.get_shape_array<eigen::Index, 5>();
    eigen::Tensor<double, 5> result{dimensions};
    variable.read(result.data());
    return result;
  }

  eigen::Tensor<std::complex<double>, 4> get_phase_matrix_data_spectral() {
      auto variable_real = group_.get_variable("phaMat_data_real");
      auto variable_imag = group_.get_variable("phaMat_data_imag");
      auto dimensions = variable_real.get_shape_array<eigen::Index, 4>();

      eigen::Tensor<float, 4> real{dimensions};
      eigen::Tensor<float, 4> imag{dimensions};
      variable_real.read(real.data());
      variable_imag.read(imag.data());
      eigen::Tensor<std::complex<double>, 4> result = imag.cast<std::complex<double>>();
      result = result * std::complex<double>(0.0, 1.0);
      result += real;
      return result;
  }

  eigen::Tensor<double, 3> get_extinction_matrix_data_gridded() {
      auto variable = group_.get_variable("extMat_data");
      auto dimensions = variable.get_shape_array<eigen::Index, 3>();
      eigen::Tensor<double, 3> result{dimensions};
      variable.read(result.data());
      return result;
  }

  eigen::Tensor<double, 3> get_extinction_matrix_data_spectral() {
      auto variable = group_.get_variable("extMat_data");
      auto dimensions = variable.get_shape_array<eigen::Index, 3>();
      eigen::Tensor<float, 3> result{dimensions};
      variable.read(result.data());
      return result.cast<double>();
  }

  eigen::Tensor<double, 3> get_absorption_vector_data_gridded() {
      auto variable = group_.get_variable("absVec_data");
      auto dimensions = variable.get_shape_array<eigen::Index, 3>();
      eigen::Tensor<double, 3> result{dimensions};
      variable.read(result.data());
      return result;
  }

  eigen::Tensor<double, 3> get_absorption_vector_data_spectral() {
      auto variable = group_.get_variable("absVec_data");
      auto dimensions = variable.get_shape_array<eigen::Index, 3>();
      eigen::Tensor<float, 3> result{dimensions};
      variable.read(result.data());
      return result.cast<double>();
  }

 private:
  Format format_;
  netcdf4::Group group_;
};

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

  ParticleFile(std::string path) : file_handle_(netcdf4::File::open(path)) {
      parse_temps_and_freqs();
  }



  std::vector<double> get_frequencies() {
      return freqs_;
  }

  std::vector<double> get_temperatures() {
      return temps_;
  }

  ScatteringData get_scattering_data(size_t i, size_t j) {
      double freq = freqs_[i];
      double temp = temps_[j];
      auto found = group_map_.find(std::make_pair(freq, temp));
      auto group = found->second.get_group("SingleScatteringData");
      return ScatteringData(group);
  }

 private:
  std::vector<double> freqs_;
  std::vector<double> temps_;
  std::map<std::pair<double, double>, netcdf4::Group> group_map_;
  netcdf4::File file_handle_;
};

// pxx :: export
class HabitFolder {

  // pxx :: hide
  void parse_files() {
    auto it = std::filesystem::directory_iterator(base_path_);
    for (auto &p : it) {
        auto match = detail::match_particle_file_name(p.path().filename());
      if (std::get<0>(match)) {
        double d_eq = std::get<1>(match);
        double d_max = std::get<2>(match);
        double mass = std::get<3>(match);
        d_eq_.push_back(d_eq);
        d_max_.push_back(d_max);
        mass_.push_back(mass);
        files_[d_eq] = base_path_ / p.path();
      }
    }
    detail::sort_by_d_eq(d_eq_, d_max_, mass_);
  }

public:

HabitFolder(std::string path) : base_path_(path) {
      parse_files();
 }

  std::vector<double> get_d_eq() {
      return d_eq_;
  }

  std::vector<double> get_d_max() {
      return d_max_;
  }

  std::vector<double> get_mass() {
      return mass_;
  }

private:
  std::filesystem::path base_path_;
    std::vector<double> d_eq_;
    std::vector<double> d_max_;
    std::vector<double> mass_;
    std::map<double, std::filesystem::path> files_;
};


}  // namespace arts_ssdb
}

#endif
