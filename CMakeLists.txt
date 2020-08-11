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

find_package(Eigen3)
find_package(pxx)

#
# External code
#

add_subdirectory(ext)
add_subdirectory(test)
add_subdirectory(doc)
add_subdirectory(python)

