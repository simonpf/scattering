cmake_minimum_required (VERSION 3.10)
project (scatlib)

#
# C++20
#

set (CMAKE_CXX_STANDARD 20)

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

set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(NetCDF)
find_package(HDF5)
find_package(Eigen3)
find_package(pxx)

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


add_library (scatlib_export INTERFACE)
target_include_directories(scatlib_export INTERFACE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/ext/shtns")
target_link_libraries(scatlib_export
  INTERFACE "${CMAKE_BINARY_DIR}/src/libscattering.a" "${CMAKE_BINARY_DIR}/ext/shtns/libshtns.a"
  )

get_target_property(scatlib_INCLUDE_DIR scatlib_export INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(scatlib_INCLUDE_DIRS scatlib_export INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(scatlib_LIBRARIES scatlib_export INTERFACE_LINK_LIBRARIES)

add_library (scatlib INTERFACE)
add_library (scatlib::scatlib ALIAS scatlib)
install (TARGETS scatlib EXPORT scatlib)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scatlibConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/scatlibConfig.cmake
  PATH_VARS scatlib_INCLUDE_DIR scatlib_LIBRARIES
  INSTALL_DESTINATION ${CMAKE_BINARY_DIR}
  NO_CHECK_REQUIRED_COMPONENTS_MACRO #
)
