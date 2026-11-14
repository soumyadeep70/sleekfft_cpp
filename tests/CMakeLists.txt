find_package(PkgConfig QUIET)

if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 QUIET IMPORTED_TARGET fftw3)
endif()

if(FFTW3_FOUND)
    add_executable(c_test_exe c_tests.c)
    target_link_libraries(c_test_exe PRIVATE sleekfft m PkgConfig::FFTW3)
    add_test(NAME CAPIValidation COMMAND c_test_exe)

    add_executable(cpp_test_exe cpp_tests.cpp)
    target_link_libraries(cpp_test_exe PRIVATE sleekfft PkgConfig::FFTW3)
    add_test(NAME CXXAPIValidation COMMAND cpp_test_exe)
else()
    message(STATUS "FFTW3 not found - skipping FFTW tests")
endif()