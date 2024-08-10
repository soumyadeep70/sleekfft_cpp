#pragma once
#include "fft_generic.h"
#include "fft_sse2.h"
#include "fft_avx.h"
#include "fft_avx_fma.h"

class FFT {
    using usize = unsigned long long;
    using f64 = double;

    f64 *wr, *wi;
    usize size;

    void (*_reserve_)(f64*&, f64*&, usize&, usize);
    void (*_fft_)(const f64*, const f64*, f64*, f64*, usize);
    void (*_ifft_)(const f64*, const f64*, f64*, f64*, usize);
    void (*_cyclic_conv_)(const f64*, const f64*, f64*, f64*, usize);

public:
    FFT() : wr(nullptr), wi(nullptr), size(0) {
        __builtin_cpu_init();
        const bool AVX_SUPPORTED = __builtin_cpu_supports("avx");
        const bool FMA_SUPPORTED = __builtin_cpu_supports("fma");
        const bool SSE2_SUPPORTED = __builtin_cpu_supports("sse2");
        if (AVX_SUPPORTED && FMA_SUPPORTED) {
            _reserve_ = FFT_AVX_FMA::reserve;
            _fft_ = FFT_AVX_FMA::fft;
            _ifft_ = FFT_AVX_FMA::ifft;
            _cyclic_conv_ = FFT_AVX_FMA::cyclic_conv;
        } else if (AVX_SUPPORTED) {
            _reserve_ = FFT_AVX::reserve;
            _fft_ = FFT_AVX::fft;
            _ifft_ = FFT_AVX::ifft;
            _cyclic_conv_ = FFT_AVX::cyclic_conv;
        } else if (SSE2_SUPPORTED) {
            _reserve_ = FFT_SSE2::reserve;
            _fft_ = FFT_SSE2::fft;
            _ifft_ = FFT_SSE2::ifft;
            _cyclic_conv_ = FFT_SSE2::cyclic_conv;
        } else {
            _reserve_ = FFT_GENERIC::reserve;
            _fft_ = FFT_GENERIC::fft;
            _ifft_ = FFT_GENERIC::ifft;
            _cyclic_conv_ = FFT_GENERIC::cyclic_conv;
        }
    }

    // computes forward transform of input data
    // input in normal order, output in bit reversed order
    void fft(f64 *real, f64 *imag, const usize k) {
        _reserve_(wr, wi, size, k);
        _fft_(wr, wi, real, imag, k);
    }

    // computes backward transform of input data
    // input in bit reversed order, output in normal order
    void ifft(f64 *real, f64 *imag, const usize k) {
        _reserve_(wr, wi, size, k);
        _ifft_(wr, wi, real, imag, k);
    }

    // perform cyclic convolution of the two real signals
    // and writes output in half of the input arrays
    void cyclic_conv(f64 *real1, f64 *real2, const usize k) {
        _reserve_(wr, wi, size, k);
        _cyclic_conv_(wr, wi, real1, real2, k);
    }

};
