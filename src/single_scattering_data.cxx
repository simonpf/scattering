#include "scatlib/single_scattering_data.h"

namespace scatlib {

//
// Gridded
//

SingleScatteringData::SingleScatteringData(
    eigen::VectorPtr<double> f_grid,
    eigen::VectorPtr<double> t_grid,
    eigen::VectorPtr<double> lon_inc,
    eigen::VectorPtr<double> lat_inc,
    eigen::VectorPtr<double> lon_scat,
    eigen::VectorPtr<double> lat_scat,
    eigen::TensorPtr<double, 7> phase_matrix,
    eigen::TensorPtr<double, 7> extinction_matrix,
    eigen::TensorPtr<double, 7> absorption_vector,
    eigen::TensorPtr<double, 7> backward_scattering_coeff,
    eigen::TensorPtr<double, 7> forward_scattering_coeff)
    : data_(new SingleScatteringDataGridded<double>(f_grid,
                                                    t_grid,
                                                    lon_inc,
                                                    lat_inc,
                                                    lon_scat,
                                                    lat_scat,
                                                    phase_matrix,
                                                    extinction_matrix,
                                                    absorption_vector,
                                                    backward_scattering_coeff,
                                                    forward_scattering_coeff)) {}

SingleScatteringData::SingleScatteringData(
    scatlib::eigen::Vector<double> f_grid,
    scatlib::eigen::Vector<double> t_grid,
    scatlib::eigen::Vector<double> lon_inc,
    scatlib::eigen::Vector<double> lat_inc,
    scatlib::eigen::Vector<double> lon_scat,
    scatlib::eigen::Vector<double> lat_scat,
    ParticleType type)
    : SingleScatteringData(
          std::make_shared<eigen::Vector<double>>(f_grid),
          std::make_shared<eigen::Vector<double>>(t_grid),
          std::make_shared<eigen::Vector<double>>(lon_inc),
          std::make_shared<eigen::Vector<double>>(lat_inc),
          std::make_shared<eigen::Vector<double>>(lon_scat),
          std::make_shared<eigen::Vector<double>>(lat_scat),
          std::make_shared<eigen::Tensor<double, 7>>(
              std::array<Index, 7>{f_grid.size(),
                                   t_grid.size(),
                                   lon_inc.size(),
                                   lat_inc.size(),
                                   lon_scat.size(),
                                   lat_scat.size(),
                                   detail::get_n_phase_matrix_elements(type)}),
          std::make_shared<eigen::Tensor<double, 7>>(std::array<Index, 7>{
              f_grid.size(),
              t_grid.size(),
              lon_inc.size(),
              lat_inc.size(),
              1,
              1,
              detail::get_n_extinction_matrix_elements(type)}),
          std::make_shared<eigen::Tensor<double, 7>>(std::array<Index, 7>{
              f_grid.size(),
              t_grid.size(),
              lon_inc.size(),
              lat_inc.size(),
              1,
              1,
              detail::get_n_absorption_vector_elements(type)}),
          std::make_shared<eigen::Tensor<double, 7>>(
              std::array<Index, 7>{f_grid.size(),
                                   t_grid.size(),
                                   lon_inc.size(),
                                   lat_inc.size(),
                                   1,
                                   1,
                                   1}),
          std::make_shared<eigen::Tensor<double, 7>>(
              std::array<Index, 7>{f_grid.size(),
                                   t_grid.size(),
                                   lon_inc.size(),
                                   lat_inc.size(),
                                   1,
                                   1,
                                   1})) {}

//
// Spectral
//

SingleScatteringData::SingleScatteringData(
    eigen::VectorPtr<double> f_grid,
    eigen::VectorPtr<double> t_grid,
    eigen::VectorPtr<double> lon_inc,
    eigen::VectorPtr<double> lat_inc,
    std::shared_ptr<sht::SHT> sht_scat,
    eigen::TensorPtr<std::complex<double>, 6> phase_matrix,
    eigen::TensorPtr<std::complex<double>, 6> extinction_matrix,
    eigen::TensorPtr<std::complex<double>, 6> absorption_vector,
    eigen::TensorPtr<std::complex<double>, 6> backward_scattering_coeff,
    eigen::TensorPtr<std::complex<double>, 6> forward_scattering_coeff)
    : data_(new SingleScatteringDataSpectral<double>(f_grid,
                                                     t_grid,
                                                     lon_inc,
                                                     lat_inc,
                                                     sht_scat,
                                                     phase_matrix,
                                                     extinction_matrix,
                                                     absorption_vector,
                                                     backward_scattering_coeff,
                                                     forward_scattering_coeff)) {}

SingleScatteringData::SingleScatteringData(
    eigen::Vector<double> f_grid,
    eigen::Vector<double> t_grid,
    eigen::Vector<double> lon_inc,
    eigen::Vector<double> lat_inc,
    sht::SHT sht_scat,
    eigen::Tensor<std::complex<double>, 6> phase_matrix,
    eigen::Tensor<std::complex<double>, 6> extinction_matrix,
    eigen::Tensor<std::complex<double>, 6> absorption_vector,
    eigen::Tensor<std::complex<double>, 6> backward_scattering_coeff,
    eigen::Tensor<std::complex<double>, 6> forward_scattering_coeff)
    : data_(new SingleScatteringDataSpectral<double>(
                std::make_shared<eigen::Vector<double>>(f_grid),
                std::make_shared<eigen::Vector<double>>(t_grid),
                std::make_shared<eigen::Vector<double>>(lon_inc),
                std::make_shared<eigen::Vector<double>>(lat_inc),
                std::make_shared<sht::SHT>(sht_scat),
                std::make_shared<eigen::Tensor<std::complex<double>, 6>>(phase_matrix),
                std::make_shared<eigen::Tensor<std::complex<double>, 6>>(extinction_matrix),
                std::make_shared<eigen::Tensor<std::complex<double>, 6>>(absorption_vector),
                std::make_shared<eigen::Tensor<std::complex<double>, 6>>(backward_scattering_coeff),
                std::make_shared<eigen::Tensor<std::complex<double>, 6>>(forward_scattering_coeff))) {}

SingleScatteringData::SingleScatteringData(
    scatlib::eigen::Vector<double> f_grid,
    scatlib::eigen::Vector<double> t_grid,
    scatlib::eigen::Vector<double> lon_inc,
    scatlib::eigen::Vector<double> lat_inc,
    Index l_max,
    ParticleType type)
    : SingleScatteringData(
          std::make_shared<eigen::Vector<double>>(f_grid),
          std::make_shared<eigen::Vector<double>>(t_grid),
          std::make_shared<eigen::Vector<double>>(lon_inc),
          std::make_shared<eigen::Vector<double>>(lat_inc),
          std::make_shared<sht::SHT>(l_max,
                                     l_max,
                                     2 * l_max + 2,
                                     2 * l_max + 2),
          std::make_shared<eigen::Tensor<std::complex<double>, 6>>(
              eigen::zeros<std::complex<double>>(
                  f_grid.size(),
                  t_grid.size(),
                  lon_inc.size(),
                  lat_inc.size(),
                  sht::SHT::calc_n_spectral_coeffs(l_max, l_max),
                  detail::get_n_phase_matrix_elements(type))),
          std::make_shared<eigen::Tensor<std::complex<double>, 6>>(
              eigen::zeros<std::complex<double>>(
                  f_grid.size(),
                  t_grid.size(),
                  lon_inc.size(),
                  lat_inc.size(),
                  1,
                  detail::get_n_extinction_matrix_elements(type))),
          std::make_shared<eigen::Tensor<std::complex<double>, 6>>(
              eigen::zeros<std::complex<double>>(
                  f_grid.size(),
                  t_grid.size(),
                  lon_inc.size(),
                  lat_inc.size(),
                  1,
                  detail::get_n_absorption_vector_elements(type))),
          std::make_shared<eigen::Tensor<std::complex<double>, 6>>(
              eigen::zeros<std::complex<double>>(
              f_grid.size(),
                                   t_grid.size(),
                                   lon_inc.size(),
                                   lat_inc.size(),
                                   1,
                      1)),
          std::make_shared<eigen::Tensor<std::complex<double>, 6>>(
          eigen::zeros<std::complex<double>>(
              f_grid.size(),
                                   t_grid.size(),
                                   lon_inc.size(),
                                   lat_inc.size(),
                                   1,
              1))) {}

}
