get_target_property(Eigen3_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
set(Eigen3_INCLUDE_DIRS ${Eigen3_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS}/unsupported)
get_target_property(netcdfhpp_INCLUDE_DIRS netcdfhpp::headers INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${netcdfhpp_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS})

#
# SHT
#

add_pxx_module(
  SOURCE ${PROJECT_SOURCE_DIR}/include/scattering/sht.h
  MODULE sht
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/ext/shtns ${PROJECT_SOURCE_DIR}/include ${Eigen3_INCLUDE_DIRS}
  )
add_dependencies(sht libshtns)
target_link_libraries(sht ${SHTNS_LIBRARY})
target_link_libraries(sht Eigen3::Eigen)
target_link_libraries(sht fftw3)
target_link_libraries(sht scattering)

#
# eigen
#

add_pxx_module(
  SOURCE ${PROJECT_SOURCE_DIR}/include/scattering/eigen.h
  MODULE eigen
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include ${Eigen3_INCLUDE_DIRS}
  )

#
# Interpolation
#

add_pxx_module(
  SOURCE ${PROJECT_SOURCE_DIR}/include/scattering/interpolation.h
  MODULE interpolation
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include ${Eigen3_INCLUDE_DIRS}
  )

#
# Scattering data field
#

add_pxx_module(
  SOURCE ${PROJECT_SOURCE_DIR}/include/scattering/scattering_data_field.h
  MODULE scattering_data_field
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/ext/shtns ${PROJECT_SOURCE_DIR}/include ${Eigen3_INCLUDE_DIRS}
  )
add_dependencies(scattering_data_field libshtns)
target_link_libraries(scattering_data_field scattering)

#
# Scattering data
#

add_pxx_module(
  SOURCE ${PROJECT_SOURCE_DIR}/include/scattering/single_scattering_data.h
  MODULE single_scattering_data
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/ext/shtns ${PROJECT_SOURCE_DIR}/include ${Eigen3_INCLUDE_DIRS}
  )
add_dependencies(single_scattering_data libshtns)
target_link_libraries(single_scattering_data scattering)

#
# arts_ssdb
#

add_pxx_module(
  SOURCE ${PROJECT_SOURCE_DIR}/include/scattering/arts_ssdb.h
  MODULE arts_ssdb
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/ext/shtns ${PROJECT_SOURCE_DIR}/include ${Eigen3_INCLUDE_DIRS} ${netcdfhpp_INCLUDE_DIRS}
  )
add_dependencies(arts_ssdb libshtns)
target_link_libraries(arts_ssdb ${NETCDF_LIBRARY} ${HDF5_LIBRARIES} scattering)

#
# integration
#

add_pxx_module(
  SOURCE ${PROJECT_SOURCE_DIR}/include/scattering/integration.h
  MODULE integration
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include ${Eigen3_INCLUDE_DIRS}
  )
target_link_libraries(integration fftw3)

#
# particle model
#

add_pxx_module(
  SOURCE ${PROJECT_SOURCE_DIR}/include/scattering/particle_habit.h
  MODULE particle_habit
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/ext/shtns ${PROJECT_SOURCE_DIR}/include ${Eigen3_INCLUDE_DIRS}
  )
add_dependencies(particle_habit libshtns)
target_link_libraries(particle_habit ${NETCDF_LIBRARY} ${HDF5_LIBRARIES} scattering)
