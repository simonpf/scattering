get_target_property(Eigen3_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(netcdfhpp_INCLUDE_DIRS netcdfhpp::headers INTERFACE_INCLUDE_DIRECTORIES)
include_directories(
  ${PROJECT_SOURCE_DIR}/ext/shtns
  ${PROJECT_SOURCE_DIR}/include
  ${netcdfhpp_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}/unsupported
  )

#
# Scattering
#

add_library(
  scattering
  sht.cxx
  single_scattering_data.cxx
  arts_ssdb.cxx)

add_dependencies(scattering libshtns)
target_link_libraries(scattering ${SHTNS_LIBRARY} fftw3)
set_property(TARGET scattering PROPERTY POSITION_INDEPENDENT_CODE ON)
