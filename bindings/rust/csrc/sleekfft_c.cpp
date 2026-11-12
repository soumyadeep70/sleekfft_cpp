#include "sleekfft_c.h"

#include "sleekfft/fft.h"

namespace {

FFT* as_fft(sleekfft* p) { return reinterpret_cast<FFT*>(p); }

// C++ exceptions must never unwind across the FFI boundary.
template <typename F> int guarded(sleekfft* self, F&& f) {
    if (self == nullptr) {
        return SLEEKFFT_ERR_NULL;
    }
    try {
        f(*as_fft(self));
        return SLEEKFFT_OK;
    } catch (...) {
        return SLEEKFFT_ERR_EXCEPTION;
    }
}

} // namespace

extern "C" {

sleekfft* sleekfft_new(void) {
    try {
        return reinterpret_cast<sleekfft*>(new FFT());
    } catch (...) {
        return nullptr;
    }
}

void sleekfft_free(sleekfft* self) { delete as_fft(self); }

int sleekfft_fft(sleekfft* self, double* real, double* imag, uint64_t k) {
    return guarded(self, [&](FFT& f) { f.fft(real, imag, k); });
}

int sleekfft_ifft(sleekfft* self, double* real, double* imag, uint64_t k) {
    return guarded(self, [&](FFT& f) { f.ifft(real, imag, k); });
}

int sleekfft_cyclic_conv(sleekfft* self, double* real1, double* real2, uint64_t k) {
    return guarded(self, [&](FFT& f) { f.cyclic_conv(real1, real2, k); });
}

} // extern "C"