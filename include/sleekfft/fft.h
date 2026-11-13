#pragma once
#include "fft_avx.h"
#include "fft_avx_fma.h"
#include "fft_generic.h"
#include "fft_sse2.h"
#include <cstddef>

class FFT {
    using f64 = double;

    f64 *wr, *wi;
    size_t size;

    void (*_plan_)(f64*&, f64*&, size_t&, size_t);
    void (*_destroy_)(f64*&, f64*&);
    void (*_fft_)(const f64*, const f64*, f64*, f64*, size_t);
    void (*_ifft_)(const f64*, const f64*, f64*, f64*, size_t);
    void (*_cyclic_conv_)(const f64*, const f64*, f64*, f64*, size_t);

  public:
    FFT() : wr(nullptr), wi(nullptr), size(0) {
        __builtin_cpu_init();
        const bool AVX_SUPPORTED = __builtin_cpu_supports("avx");
        const bool FMA_SUPPORTED = __builtin_cpu_supports("fma");
        const bool SSE2_SUPPORTED = __builtin_cpu_supports("sse2");
        if (AVX_SUPPORTED && FMA_SUPPORTED) {
            _plan_ = FFT_AVX_FMA::plan;
            _destroy_ = FFT_AVX_FMA::destroy;
            _fft_ = FFT_AVX_FMA::fft;
            _ifft_ = FFT_AVX_FMA::ifft;
            _cyclic_conv_ = FFT_AVX_FMA::cyclic_conv;
        } else if (AVX_SUPPORTED) {
            _plan_ = FFT_AVX::plan;
            _destroy_ = FFT_AVX::destroy;
            _fft_ = FFT_AVX::fft;
            _ifft_ = FFT_AVX::ifft;
            _cyclic_conv_ = FFT_AVX::cyclic_conv;
        } else if (SSE2_SUPPORTED) {
            _plan_ = FFT_SSE2::plan;
            _destroy_ = FFT_SSE2::destroy;
            _fft_ = FFT_SSE2::fft;
            _ifft_ = FFT_SSE2::ifft;
            _cyclic_conv_ = FFT_SSE2::cyclic_conv;
        } else {
            _plan_ = FFT_GENERIC::plan;
            _destroy_ = FFT_GENERIC::destroy;
            _fft_ = FFT_GENERIC::fft;
            _ifft_ = FFT_GENERIC::ifft;
            _cyclic_conv_ = FFT_GENERIC::cyclic_conv;
        }
    }

    ~FFT() { _destroy_(wr, wi); }

    // computes forward transform of input data
    // input in normal order, output in bit reversed order
    void fft(f64* real, f64* imag, const size_t k) {
        _plan_(wr, wi, size, k);
        _fft_(wr, wi, real, imag, k);
    }

    // computes backward transform of input data
    // input in bit reversed order, output in normal order
    void ifft(f64* real, f64* imag, const size_t k) {
        _plan_(wr, wi, size, k);
        _ifft_(wr, wi, real, imag, k);
    }

    // perform cyclic convolution of the two real signals
    // and writes output in half of the input arrays
    void cyclic_conv(f64* real1, f64* real2, const size_t k) {
        _plan_(wr, wi, size, k);
        _cyclic_conv_(wr, wi, real1, real2, k);
    }
};
