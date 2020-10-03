get_target_property(Eigen3_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)

#
# Scattering
#

add_library(scattering sht.cxx single_scattering_data.cxx)
target_include_directories(scattering
  PUBLIC "${CMAKE_SOURCE_DIR}/ext/shtns"
  PUBLIC "${CMAKE_SOURCE_DIR}/include"
  PUBLIC ${Eigen3_INCLUDE_DIRS}
  PUBLIC ${Eigen3_INCLUDE_DIRS}/unsupported)
add_dependencies(scattering libshtns)
target_link_libraries(scattering ${SHTNS_LIBRARY} fftw3)
set_property(TARGET scattering PROPERTY POSITION_INDEPENDENT_CODE ON)
