get_target_property(Eigen3_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
set(Eigen3_INCLUDE_DIRS ${Eigen3_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS}/unsupported)
message(WARNING "EIGEN: ${Eigen3_INCLUDE_DIRS}")

#
# SHT
#

add_pxx_module(
  SOURCE ${PROJECT_SOURCE_DIR}/include/scatlib/sht.h
  MODULE sht
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/ext/shtns ${PROJECT_SOURCE_DIR}/include ${Eigen3_INCLUDE_DIRS}
  )
add_dependencies(sht libshtns)
target_link_libraries(sht ${SHTNS_LIBRARY})
target_link_libraries(sht Eigen3::Eigen)
target_link_libraries(sht fftw3)

#
# eigen
#

add_pxx_module(
  SOURCE ${PROJECT_SOURCE_DIR}/include/scatlib/eigen.h
  MODULE eigen
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include ${Eigen3_INCLUDE_DIRS}
  )

#
# Interpolation
#

add_pxx_module(
  SOURCE ${PROJECT_SOURCE_DIR}/include/scatlib/interpolation.h
  MODULE interpolation
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include ${Eigen3_INCLUDE_DIRS}
  )

#
# Scattering data field
#

add_pxx_module(
  SOURCE ${PROJECT_SOURCE_DIR}/include/scatlib/scattering_data_field.h
  MODULE scattering_data_field
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/ext/shtns ${PROJECT_SOURCE_DIR}/include ${Eigen3_INCLUDE_DIRS}
  )
add_dependencies(scattering_data_field libshtns eigen interpolation)
target_link_libraries(scattering_data_field ${SHTNS_LIBRARY})
target_link_libraries(scattering_data_field fftw3)

#
# Scattering data
#

add_pxx_module(
  SOURCE ${PROJECT_SOURCE_DIR}/include/scatlib/scattering_data.h
  MODULE scattering_data
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/ext/shtns ${PROJECT_SOURCE_DIR}/include ${Eigen3_INCLUDE_DIRS}
  )
add_dependencies(scattering_data libshtns)
target_link_libraries(scattering_data ${SHTNS_LIBRARY})
target_link_libraries(scattering_data fftw3)
