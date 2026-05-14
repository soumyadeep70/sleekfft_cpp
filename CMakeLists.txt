cmake_minimum_required(VERSION 3.28)
project(sleekfft_cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
add_subdirectory(sleekfft)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED IMPORTED_TARGET fftw3)

add_executable(test tests/main.cpp)
target_link_libraries(test PRIVATE SleekFFT PkgConfig::FFTW)