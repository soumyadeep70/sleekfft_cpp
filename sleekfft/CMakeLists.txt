cmake_minimum_required(VERSION 3.28)
project(sleekfft_cpp)

set(CMAKE_CXX_STANDARD 11)

add_library(SleekFFT INTERFACE fft.h fft_generic.h fft_sse2.h fft_avx.h fft_avx_fma.h)
target_include_directories(SleekFFT INTERFACE .)