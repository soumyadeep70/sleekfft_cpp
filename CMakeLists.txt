cmake_minimum_required(VERSION 3.28)
project(sleekfft LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

add_library(sleekfft INTERFACE)
target_include_directories(sleekfft INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(sleekfft INTERFACE cxx_std_23)

option(ENABLE_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()