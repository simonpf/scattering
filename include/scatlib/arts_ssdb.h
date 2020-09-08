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

class ParticleGroup {
 public:
  ParticleGroup(netcdf4::Group group) : group_(group) {}

  eigen::Tensor<double, 7> get_phase_matrix_data() {
    auto variable = group_.get_variable("phase_matrix_data");
    auto shape = variable.shape();
    std::array<eigen::Index, 7> dimensions;
    for (int i = 0; i < 7; ++i) {
      dimensions[i] = shape[i];
    }

    eigen::Tensor<double, 7> result{dimensions};
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
    for (auto &name : group_names) {
      auto freq_and_temp = detail::match_temp_and_freq(name);
      freqs_.insert(std::get<0>(freq_and_temp));
      temps_.insert(std::get<1>(freq_and_temp));
      group_map_[freq_and_temp] = file_handle.get_group(name);
    }
    std::sort(freqs_.begin(), freqs_.end());
    std::sort(temps_.begin(), temps_.end());
  }


 public:

  ParticleFile(std::string path) : file_handle_(netcdf4::File::open(path)) {
      parse_temps_and_freqs();
  }

  std::set<double> get_frequencies() {
      return freqs_;
  }

  std::set<double> get_temperatures() {
      return temps_;
  }

 private:
  std::set<double> freqs_;
  std::set<double> temps_;
  std::map<std::pair<double, double>, netcdf4::Group> group_map_;
  netcdf4::File file_handle_;
};


}  // namespace arts_ssdb
}

#endif
