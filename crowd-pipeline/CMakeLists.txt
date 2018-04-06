cmake_minimum_required (VERSION 3.9)
project(LIBMRF VERSION 1.0 DESCRIPTION "mrf library")
enable_language (CXX)
set(CMAKE_CXX_STANDARD 11)

# make sure that the default is a RELEASE
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)
endif()

add_library(mrf SHARED
  ${CMAKE_SOURCE_DIR}/image.hpp
  ${CMAKE_SOURCE_DIR}/mrflib.cpp
)

set_property(TARGET mrf PROPERTY POSITION_INDEPENDENT_CODE ON)
install(TARGETS mrf DESTINATION ${CMAKE_SOURCE_DIR}/lib/)

