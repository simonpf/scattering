#ifndef __SCATLIB_ARTS_SSDB__
#define __SCATLIB_ARTS_SSDB__

#include "netcdfpp.hpp"
#include <regex>
#include <set>
#include <utility>

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

}

// pxx :: export
class ScatteringData {
 public:
  ScatteringData(netcdf4::Group group) : group_(group) {}

  eigen::Tensor<double, 7> get_phase_matrix_data() {
    auto variable = group_.get_variable("phaMat_data");
    auto dimensions = variable.get_shape_array<eigen::Index, 5>();
    std::array<eigen::Index, 7> full_dimensions = {1, 1};
    for (int i = 0; i < 5; ++i) {
        full_dimensions[i + 2] = dimensions[i];
    }
    eigen::Tensor<double, 7> result{full_dimensions};
    variable.read(result.data());
    return result;
  }

 private:
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


}  // namespace arts_ssdb
}

#endif
