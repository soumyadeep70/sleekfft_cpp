cmake_minimum_required(VERSION 3.28)
project(sleekfft LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(sleekfft INTERFACE)
target_include_directories(sleekfft INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(sleekfft INTERFACE c_std_11 cxx_std_11)

option(ENABLE_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()