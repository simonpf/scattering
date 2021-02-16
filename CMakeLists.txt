cmake_minimum_required (VERSION 3.10)
project (scattering)

#
# C++20
#

set (CMAKE_CXX_STANDARD 17)

#
# Build flags
#

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -finline-functions")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

#
# Find required packages
#

set (CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(NetCDF)
find_package(HDF5)
find_package(Eigen3)
find_package(pxx)
find_package(netcdfhpp)

#
# External code
#

add_subdirectory(ext)
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(doc)
add_subdirectory(python)

#
# Install CMake package.
#

include (CMakePackageConfigHelpers)

add_library (scattering_export INTERFACE)
target_include_directories(
  scattering_export
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/ext/shtns>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:ext/shtns>
  )
target_link_libraries(
  scattering_export
  INTERFACE "${PROJECT_BINARY_DIR}/src/libscattering.a" "${PROJECT_BINARY_DIR}/ext/shtns/libshtns.a"
  )

set_target_properties (scattering_export PROPERTIES EXPORT_NAME scattering)

get_target_property(scattering_INCLUDE_DIR scattering_export INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(scattering_INCLUDE_DIRS scattering_export INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(scattering_LIBRARIES scattering_export INTERFACE_LINK_LIBRARIES)

install (TARGETS scattering_export EXPORT scattering)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scatteringConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/scatteringConfig.cmake
  PATH_VARS scattering_INCLUDE_DIR scattering_LIBRARIES
  INSTALL_DESTINATION ${PROJECT_BINARY_DIR}
  NO_CHECK_REQUIRED_COMPONENTS_MACRO #
)

message(WARNING ${netcdfhpp_DIR})
export (TARGETS scattering_export NAMESPACE scattering:: FILE scatteringTargets.cmake)
export (PACKAGE scattering)
install (EXPORT scattering NAMESPACE scattering:: DESTINATION ${PROJECT_BINARY_DIR})

message(WARNING SCATTERING)
