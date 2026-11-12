//! Raw, unsafe declarations mirroring `csrc/sleekfft_c.h`.

use std::{marker::PhantomData, marker::PhantomPinned, os::raw::c_int};

/// Opaque handle to the C++ `FFT` object.
#[repr(C)]
pub struct Raw {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

pub const SLEEKFFT_OK: c_int = 0;

unsafe extern "C" {
    pub unsafe fn sleekfft_new() -> *mut Raw;
    pub unsafe fn sleekfft_free(this: *mut Raw);
    pub unsafe fn sleekfft_fft(this: *mut Raw, real: *mut f64, imag: *mut f64, k: u64) -> c_int;
    pub unsafe fn sleekfft_ifft(this: *mut Raw, real: *mut f64, imag: *mut f64, k: u64) -> c_int;
    pub unsafe fn sleekfft_cyclic_conv(
        this: *mut Raw,
        real1: *mut f64,
        real2: *mut f64,
        k: u64,
    ) -> c_int;
}
