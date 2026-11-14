#ifndef SLEEKFFT_H
#define SLEEKFFT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "common.h"
#include "scalar.h"
#include "x64_avx.h"
#include "x64_avx_fma.h"
#include "x64_sse2.h"
#include <stdio.h>

typedef enum {
    SLEEKFFT_OK = 0,
    SLEEKFFT_INVALID_CONTEXT = -1,
    SLEEKFFT_ERR_BAD_INPUT = -3,
} sleekfft_status;

typedef struct sleekfft_context {
    void (*_reserve_twiddles_)(twiddle_table*, size_t);
    void (*_forward_transform_)(const twiddle_table*, f64*, f64*, size_t);
    void (*_inverse_transform_)(const twiddle_table*, f64*, f64*, size_t);
    void (*_cyclic_conv_)(const twiddle_table*, f64*, f64*, size_t);

    twiddle_table* table;
} sleekfft_context;

sleekfft_context* sleekfft_create_context() {
    sleekfft_context* context = (sleekfft_context*)malloc(sizeof(sleekfft_context));
    context->table = NULL;
    __builtin_cpu_init();
    const int AVX_SUPPORTED = __builtin_cpu_supports("avx");
    const int FMA_SUPPORTED = __builtin_cpu_supports("fma");
    const int SSE2_SUPPORTED = __builtin_cpu_supports("sse2");
    if (AVX_SUPPORTED && FMA_SUPPORTED) {
        context->_reserve_twiddles_ = x64_avx_fma_reserve_twiddles;
        context->_forward_transform_ = x64_avx_fma_forward_transform;
        context->_inverse_transform_ = x64_avx_fma_inverse_transform;
        context->_cyclic_conv_ = x64_avx_fma_cyclic_conv;
    } else if (AVX_SUPPORTED) {
        context->_reserve_twiddles_ = x64_avx_reserve_twiddles;
        context->_forward_transform_ = x64_avx_forward_transform;
        context->_inverse_transform_ = x64_avx_inverse_transform;
        context->_cyclic_conv_ = x64_avx_cyclic_conv;
    } else if (SSE2_SUPPORTED) {
        context->_reserve_twiddles_ = x64_sse2_reserve_twiddles;
        context->_forward_transform_ = x64_sse2_forward_transform;
        context->_inverse_transform_ = x64_sse2_inverse_transform;
        context->_cyclic_conv_ = x64_sse2_cyclic_conv;
    } else {
        context->_reserve_twiddles_ = scalar_reserve_twiddles;
        context->_forward_transform_ = scalar_forward_transform;
        context->_inverse_transform_ = scalar_inverse_transform;
        context->_cyclic_conv_ = scalar_cyclic_conv;
    }
    return context;
}

int sleekfft_destroy_context(const sleekfft_context* context) {
    if (context) {
        if (context->table) {
            free(context->table->wr);
            free(context->table->wi);
            free(context->table);
        }
        free((void*)context);
    }
    return SLEEKFFT_OK;
}

int sleekfft_reserve_twiddles(sleekfft_context* context, const size_t k) {
    if (!context)
        return SLEEKFFT_INVALID_CONTEXT;
    if (context->table == NULL) {
        context->table = (twiddle_table*)malloc(sizeof(twiddle_table));
        context->table->wr = NULL;
        context->table->wi = NULL;
        context->table->size = 0;
    }
    context->_reserve_twiddles_(context->table, k);
    return SLEEKFFT_OK;
}

int sleekfft_forward_transform(sleekfft_context* context, f64* xr, f64* xi, const size_t k) {
    if (!context)
        return SLEEKFFT_INVALID_CONTEXT;
    sleekfft_reserve_twiddles(context, k);
    context->_forward_transform_(context->table, xr, xi, k);
    return SLEEKFFT_OK;
}

int sleekfft_inverse_transform(sleekfft_context* context, f64* xr, f64* xi, const size_t k) {
    if (!context)
        return SLEEKFFT_INVALID_CONTEXT;
    sleekfft_reserve_twiddles(context, k);
    context->_inverse_transform_(context->table, xr, xi, k);
    return SLEEKFFT_OK;
}

int sleekfft_cyclic_conv(sleekfft_context* context, f64* xr, f64* xi, const size_t k) {
    if (!context)
        return SLEEKFFT_INVALID_CONTEXT;
    sleekfft_reserve_twiddles(context, k);
    context->_cyclic_conv_(context->table, xr, xi, k);
    return SLEEKFFT_OK;
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

class Sleekfft {
    sleekfft_context* context;

  public:
    Sleekfft() { context = sleekfft_create_context(); }

    ~Sleekfft() { sleekfft_destroy_context(context); }

    // computes forward transform of input data
    // input in normal order, output in bit reversed order
    void forward_transform(f64* real, f64* imag, const size_t k) {
        sleekfft_forward_transform(context, real, imag, k);
    }

    // computes backward transform of input data
    // input in bit reversed order, output in normal order
    void inverse_transform(f64* real, f64* imag, const size_t k) {
        sleekfft_inverse_transform(context, real, imag, k);
    }

    // perform cyclic convolution of the two real signals
    // and writes output in half of the input arrays
    void cyclic_conv(f64* real1, f64* real2, const size_t k) {
        sleekfft_cyclic_conv(context, real1, real2, k);
    }
};

#endif // __cplusplus

#endif // SLEEKFFT_H