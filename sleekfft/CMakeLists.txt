cmake_minimum_required(VERSION 3.28)
project(sleekfft)

set(CMAKE_CXX_STANDARD 23)

add_library(sleekfft INTERFACE fft.h fft_generic.h fft_sse2.h fft_avx.h fft_avx_fma.h)
target_include_directories(sleekfft INTERFACE .)