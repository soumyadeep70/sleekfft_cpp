cmake_minimum_required(VERSION 3.28)
project(sleekfft_cpp)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(sleekfft)

set(fftw_include ${PROJECT_SOURCE_DIR}/external/fftw3)
set(fftw_lib ${PROJECT_SOURCE_DIR}/external/fftw3/libfftw3.a)

add_executable(test test.cpp)
target_link_libraries(test PRIVATE SleekFFT)
target_include_directories(test PRIVATE ${fftw_include})
target_link_libraries(test PRIVATE ${fftw_lib})
