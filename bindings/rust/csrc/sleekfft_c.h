#ifndef SLEEKFFT_C_H
#define SLEEKFFT_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle wrapping a heap-allocated C++ `FFT` object. */
typedef struct sleekfft sleekfft;

enum {
    SLEEKFFT_OK = 0,
    SLEEKFFT_ERR_EXCEPTION = 1, /* a C++ exception was caught at the boundary */
    SLEEKFFT_ERR_NULL = 2       /* handle was NULL */
};

/* Returns NULL if construction fails. */
sleekfft* sleekfft_new(void);
void sleekfft_free(sleekfft* self);

/* In-place forward transform: normal order in, bit-reversed order out. */
int sleekfft_fft(sleekfft* self, double* real, double* imag, uint64_t k);

/* In-place backward transform: bit-reversed order in, normal order out. */
int sleekfft_ifft(sleekfft* self, double* real, double* imag, uint64_t k);

/* Cyclic convolution of two real signals; output is written into the inputs. */
int sleekfft_cyclic_conv(sleekfft* self, double* real1, double* real2, uint64_t k);

#ifdef __cplusplus
}
#endif

#endif /* SLEEKFFT_C_H */