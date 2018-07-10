cmake_minimum_required (VERSION 3.10)
project (MRF)
set(target_name "MRF")
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

find_package(Matlab COMPONENTS MX_LIBRARY REQUIRED)
#find_package(OpenMP REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/MRF/jsoncons/include)

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
#  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4244 /wd4146 /wd4305 /wd4334 /wd4267")
endif()

add_library(mrf_core ${CMAKE_SOURCE_DIR}/${target_name}/${target_name}_core.cpp)
set_property(TARGET mrf_core PROPERTY POSITION_INDEPENDENT_CODE ON)
add_executable(physycom_mrf ${CMAKE_SOURCE_DIR}/${target_name}/${target_name}_physycom.cpp)
target_link_libraries(physycom_mrf mrf_core)
install(TARGETS physycom_mrf DESTINATION ${CMAKE_SOURCE_DIR}/${target_name}/)

matlab_add_mex(NAME ${target_name} SRC ${target_name}/${target_name}.cpp LINK_TO mrf_core)
install(TARGETS ${target_name} DESTINATION ${CMAKE_SOURCE_DIR}/${target_name}/)
