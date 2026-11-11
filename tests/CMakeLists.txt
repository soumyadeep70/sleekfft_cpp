find_package(PkgConfig QUIET)

if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 QUIET IMPORTED_TARGET fftw3)
endif()

if(FFTW3_FOUND)
    add_executable(test_exe main.cpp)
    target_link_libraries(test_exe PRIVATE sleekfft PkgConfig::FFTW3)
    add_test(NAME fftw COMMAND test_exe)
else()
    message(STATUS "FFTW3 not found - skipping FFTW tests")
endif()