//! Safe Rust bindings for the sleekfft C++ library.
//!
//! `k` is the base-2 logarithm of the transform length, so every slice passed
//! to a transform must hold at least `2^k` elements.

mod ffi;

use std::{error::Error as StdError, fmt, ptr::NonNull};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The C++ object could not be constructed.
    Allocation,
    /// `2^k` does not fit in `usize`.
    SizeOverflow { k: u32 },
    /// A slice holds fewer than `2^k` elements.
    SliceTooShort {
        name: &'static str,
        needed: usize,
        got: usize,
    },
    /// The C++ side threw an exception (caught at the FFI boundary).
    Backend(i32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Allocation => write!(f, "failed to construct the FFT object"),
            Error::SizeOverflow { k } => write!(f, "2^{k} does not fit in usize"),
            Error::SliceTooShort { name, needed, got } => {
                write!(f, "`{name}` has {got} elements, but {needed} are required")
            }
            Error::Backend(code) => write!(f, "FFT backend failed with status {code}"),
        }
    }
}

impl StdError for Error {}

/// Owning handle to a C++ `FFT` object.
///
/// `Send` but not `Sync`: the C++ methods are non-const and may lazily build
/// twiddle tables, so concurrent use of one instance is not safe.
pub struct Fft {
    raw: NonNull<ffi::Raw>,
}

// SAFETY: the C++ object owns its buffers and has no thread affinity.
unsafe impl Send for Fft {}

/// Number of elements a slice must hold for a transform of size `2^k`.
fn required_len(k: u32) -> Result<usize, Error> {
    1usize.checked_shl(k).ok_or(Error::SizeOverflow { k })
}

fn check_len(name: &'static str, slice: &[f64], needed: usize) -> Result<(), Error> {
    if slice.len() < needed {
        Err(Error::SliceTooShort {
            name,
            needed,
            got: slice.len(),
        })
    } else {
        Ok(())
    }
}

fn status(code: std::os::raw::c_int) -> Result<(), Error> {
    if code == ffi::SLEEKFFT_OK {
        Ok(())
    } else {
        Err(Error::Backend(code))
    }
}

impl Fft {
    pub fn new() -> Result<Self, Error> {
        // SAFETY: no preconditions; returns NULL on failure.
        let ptr = unsafe { ffi::sleekfft_new() };
        NonNull::new(ptr)
            .map(|raw| Self { raw })
            .ok_or(Error::Allocation)
    }

    /// Forward transform in place: normal-order input, bit-reversed output.
    pub fn fft(&mut self, real: &mut [f64], imag: &mut [f64], k: u32) -> Result<(), Error> {
        let n = required_len(k)?;
        check_len("real", real, n)?;
        check_len("imag", imag, n)?;
        // SAFETY: handle is valid; both slices are distinct (`&mut`) and hold >= 2^k elements.
        status(unsafe {
            ffi::sleekfft_fft(
                self.raw.as_ptr(),
                real.as_mut_ptr(),
                imag.as_mut_ptr(),
                k.into(),
            )
        })
    }

    /// Backward transform in place: bit-reversed input, normal-order output.
    pub fn ifft(&mut self, real: &mut [f64], imag: &mut [f64], k: u32) -> Result<(), Error> {
        let n = required_len(k)?;
        check_len("real", real, n)?;
        check_len("imag", imag, n)?;
        // SAFETY: as in `fft`.
        status(unsafe {
            ffi::sleekfft_ifft(
                self.raw.as_ptr(),
                real.as_mut_ptr(),
                imag.as_mut_ptr(),
                k.into(),
            )
        })
    }

    /// Cyclic convolution of two real signals; the result is written into the inputs
    /// (see the C++ documentation for exactly where).
    pub fn cyclic_conv(
        &mut self,
        real1: &mut [f64],
        real2: &mut [f64],
        k: u32,
    ) -> Result<(), Error> {
        let n = required_len(k)?;
        check_len("real1", real1, n)?;
        check_len("real2", real2, n)?;
        // SAFETY: as in `fft`.
        status(unsafe {
            ffi::sleekfft_cyclic_conv(
                self.raw.as_ptr(),
                real1.as_mut_ptr(),
                real2.as_mut_ptr(),
                k.into(),
            )
        })
    }
}

impl Drop for Fft {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `sleekfft_new` and is freed exactly once.
        unsafe { ffi::sleekfft_free(self.raw.as_ptr()) }
    }
}

/// Reorders the first `2^k` elements of `real` and `imag` by bit-reversed index.
///
/// [`Fft::fft`] emits its output in bit-reversed order; running this afterwards
/// yields natural order. The permutation is its own inverse.
pub fn bit_reverse_permute(real: &mut [f64], imag: &mut [f64], k: u32) -> Result<(), Error> {
    let n = required_len(k)?;
    check_len("real", real, n)?;
    check_len("imag", imag, n)?;
    if k == 0 {
        return Ok(());
    }
    let shift = usize::BITS - k;
    for i in 0..n {
        let j = i.reverse_bits() >> shift;
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fftw::{
        array::AlignedVec,
        plan::{C2CPlan, C2CPlan64},
        types::{Flag, Sign, c64},
    };
    use std::time::{Duration, Instant};

    /// Relative L2 tolerance per butterfly stage (`k` stages in total).
    /// Typical FFT error is a few 1e-16; this leaves headroom while still
    /// catching any real defect, which shows up many orders of magnitude higher.
    const REL_TOL_PER_STAGE: f64 = 1e-14;

    fn tol(k: u32) -> f64 {
        REL_TOL_PER_STAGE * f64::from(k.max(1))
    }

    // ---------------------------------------------------------------- helpers

    /// Deterministic splitmix64 generator, so failures are reproducible.
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
            let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            lo + (hi - lo) * u
        }
    }

    /// Random complex signal of length `2^k` with parts uniform in [1, 10000],
    /// matching the C++ benchmark.
    fn random_signal(k: u32, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let n = 1usize << k;
        let mut rng = Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ u64::from(k));
        let re: Vec<f64> = (0..n).map(|_| rng.uniform(1.0, 10_000.0)).collect();
        let im: Vec<f64> = (0..n).map(|_| rng.uniform(1.0, 10_000.0)).collect();
        (re, im)
    }

    struct Diff {
        max_abs: f64,
        rel_l2: f64,
    }

    /// Compares `(ar, ai)` against the reference `(br, bi)`.
    fn diff(ar: &[f64], ai: &[f64], br: &[f64], bi: &[f64]) -> Diff {
        let (mut max_abs, mut err2, mut ref2) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..br.len() {
            let dr = (ar[i] - br[i]).abs();
            let di = (ai[i] - bi[i]).abs();
            max_abs = max_abs.max(dr).max(di);
            err2 += dr * dr + di * di;
            ref2 += br[i] * br[i] + bi[i] * bi[i];
        }
        Diff {
            max_abs,
            rel_l2: (err2 / ref2).sqrt(),
        }
    }

    fn split(v: &[c64]) -> (Vec<f64>, Vec<f64>) {
        (
            v.iter().map(|c| c.re).collect(),
            v.iter().map(|c| c.im).collect(),
        )
    }

    fn fftw_plan(n: usize) -> C2CPlan64 {
        C2CPlan::aligned(&[n], Sign::Forward, Flag::ESTIMATE).expect("failed to create FFTW plan")
    }

    /// Forward FFTW transform in natural order.
    fn fftw_forward(re: &[f64], im: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = re.len();
        let mut plan = fftw_plan(n);
        let mut input: AlignedVec<c64> = AlignedVec::new(n);
        let mut output: AlignedVec<c64> = AlignedVec::new(n);
        for i in 0..n {
            input[i] = c64::new(re[i], im[i]);
        }
        plan.c2c(&mut input, &mut output)
            .expect("FFTW execution failed");
        split(&output)
    }

    /// Forward sleekfft transform, reordered to natural order.
    fn sleek_forward(fft: &mut Fft, re: &[f64], im: &[f64], k: u32) -> (Vec<f64>, Vec<f64>) {
        let (mut r, mut i) = (re.to_vec(), im.to_vec());
        fft.fft(&mut r, &mut i, k).unwrap();
        bit_reverse_permute(&mut r, &mut i, k).unwrap();
        (r, i)
    }

    fn assert_matches_fftw(fft: &mut Fft, k: u32, seed: u64) {
        let (re, im) = random_signal(k, seed);
        let (er, ei) = fftw_forward(&re, &im);
        let (gr, gi) = sleek_forward(fft, &re, &im, k);
        let d = diff(&gr, &gi, &er, &ei);
        // `!(x <= tol)` also rejects NaN.
        assert!(
            d.rel_l2 <= tol(k),
            "k={k}: relative L2 error {:.3e} (max |diff| {:.3e}) exceeds {:.1e}",
            d.rel_l2,
            d.max_abs,
            tol(k)
        );
    }

    fn best_of(reps: usize, mut run: impl FnMut() -> Duration) -> f64 {
        (0..reps)
            .map(|_| run().as_secs_f64())
            .fold(f64::INFINITY, f64::min)
    }

    // ------------------------------------------------------------ API checks

    #[test]
    fn rejects_short_slices() {
        let mut f = Fft::new().unwrap();
        let mut re = vec![0.0; 8];
        let mut im = vec![0.0; 4];
        assert_eq!(
            f.fft(&mut re, &mut im, 3),
            Err(Error::SliceTooShort {
                name: "imag",
                needed: 8,
                got: 4
            })
        );
        assert_eq!(
            f.ifft(&mut im, &mut re, 3),
            Err(Error::SliceTooShort {
                name: "real",
                needed: 8,
                got: 4
            })
        );
        assert_eq!(
            bit_reverse_permute(&mut re, &mut im, 3),
            Err(Error::SliceTooShort {
                name: "imag",
                needed: 8,
                got: 4
            })
        );
    }

    #[test]
    fn rejects_overflowing_k() {
        let mut f = Fft::new().unwrap();
        assert_eq!(
            f.fft(&mut [], &mut [], 64),
            Err(Error::SizeOverflow { k: 64 })
        );
    }

    #[test]
    fn bit_reverse_known_values() {
        let mut re: Vec<f64> = (0..8).map(f64::from).collect();
        let mut im = re.clone();
        bit_reverse_permute(&mut re, &mut im, 3).unwrap();
        assert_eq!(re, [0.0, 4.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0]);
        assert_eq!(im, re);
        // Involution: applying it twice restores the original order.
        bit_reverse_permute(&mut re, &mut im, 3).unwrap();
        assert_eq!(re, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    // ------------------------------------------------- correctness vs FFTW

    #[test]
    fn matches_fftw_k1_to_k20() {
        let mut fft = Fft::new().unwrap();
        for k in 1..=20 {
            assert_matches_fftw(&mut fft, k, 1);
        }
    }

    /// Same sweep as the C++ test (up to 2^25); needs a few GB of RAM.
    #[test]
    #[ignore = "slow and memory-hungry: cargo test -- --ignored"]
    fn matches_fftw_full_range() {
        for k in 1..=25 {
            let mut fft = Fft::new().unwrap();
            assert_matches_fftw(&mut fft, k, 2);
        }
    }

    #[test]
    fn matches_fftw_for_several_seeds() {
        let mut fft = Fft::new().unwrap();
        for seed in 0..16 {
            for k in [1, 2, 3, 5, 8, 11] {
                assert_matches_fftw(&mut fft, k, seed);
            }
        }
    }

    /// One instance reused with different sizes in non-monotonic order
    /// (catches stale or wrongly cached twiddle tables).
    #[test]
    fn reuse_one_instance_across_sizes() {
        let mut fft = Fft::new().unwrap();
        for (seed, k) in [3u32, 10, 4, 12, 3, 1, 15, 10, 2, 15]
            .into_iter()
            .enumerate()
        {
            assert_matches_fftw(&mut fft, k, seed as u64);
        }
    }

    /// Slices starting at every 8-byte offset must give the same answer as an
    /// aligned slice. A kernel that assumes 32-byte alignment would crash or
    /// diverge here.
    #[test]
    fn unaligned_slices_match_aligned() {
        let k = 12;
        let n = 1usize << k;
        let (re, im) = random_signal(k, 7);
        let mut fft = Fft::new().unwrap();

        let (mut want_r, mut want_i) = (re.clone(), im.clone());
        fft.fft(&mut want_r, &mut want_i, k).unwrap();

        for offset in 1..4 {
            let mut br = vec![0.0; n + offset];
            let mut bi = vec![0.0; n + offset];
            br[offset..].copy_from_slice(&re);
            bi[offset..].copy_from_slice(&im);
            fft.fft(&mut br[offset..], &mut bi[offset..], k).unwrap();
            let d = diff(&br[offset..], &bi[offset..], &want_r, &want_i);
            assert!(
                d.rel_l2 <= tol(k),
                "offset {offset}: relative L2 error {:.3e}",
                d.rel_l2
            );
        }
    }

    /// The transform must only touch the first `2^k` elements.
    #[test]
    fn does_not_write_past_length() {
        let k = 6;
        let n = 1usize << k;
        let (mut re, mut im) = random_signal(k, 11);
        re.extend([-1.0; 16]);
        im.extend([-2.0; 16]);
        let mut fft = Fft::new().unwrap();
        fft.fft(&mut re, &mut im, k).unwrap();
        assert!(
            re[n..].iter().all(|&x| x == -1.0),
            "wrote past the end of `real`"
        );
        assert!(
            im[n..].iter().all(|&x| x == -2.0),
            "wrote past the end of `imag`"
        );
    }

    // ------------------------------------------ analytic and round-trip checks

    #[test]
    fn impulse_gives_flat_spectrum() {
        let mut fft = Fft::new().unwrap();
        for k in 1..=16 {
            let n = 1usize << k;
            let mut re = vec![0.0; n];
            let mut im = vec![0.0; n];
            re[0] = 1.0;
            fft.fft(&mut re, &mut im, k).unwrap();
            // A unit impulse at index 0 transforms to all ones, in any bin order.
            assert!(
                re.iter().all(|&x| (x - 1.0).abs() < 1e-12),
                "k={k}: real part not flat"
            );
            assert!(
                im.iter().all(|&x| x.abs() < 1e-12),
                "k={k}: imaginary part not zero"
            );
        }
    }

    #[test]
    fn constant_signal_only_has_dc() {
        let mut fft = Fft::new().unwrap();
        for k in 1..=16 {
            let n = 1usize << k;
            let (c_re, c_im) = (3.5, -2.0);
            let mut re = vec![c_re; n];
            let mut im = vec![c_im; n];
            fft.fft(&mut re, &mut im, k).unwrap();
            // The DC bin is index 0 in both natural and bit-reversed order.
            let scale = n as f64 * 4.0;
            assert!(
                (re[0] - n as f64 * c_re).abs() < 1e-12 * scale,
                "k={k}: DC real"
            );
            assert!(
                (im[0] - n as f64 * c_im).abs() < 1e-12 * scale,
                "k={k}: DC imag"
            );
            assert!(
                re[1..].iter().all(|&x| x.abs() < 1e-12 * scale),
                "k={k}: leakage in real"
            );
            assert!(
                im[1..].iter().all(|&x| x.abs() < 1e-12 * scale),
                "k={k}: leakage in imag"
            );
        }
    }

    /// `ifft(fft(x))` must equal `x` up to a constant factor of 1 or `n`,
    /// depending on whether the library normalises the inverse.
    #[test]
    fn roundtrip_recovers_input() {
        let mut fft = Fft::new().unwrap();
        for k in 1..=18 {
            let n = 1usize << k;
            let (re0, im0) = random_signal(k, 5);
            let (mut re, mut im) = (re0.clone(), im0.clone());
            fft.fft(&mut re, &mut im, k).unwrap();
            fft.ifft(&mut re, &mut im, k).unwrap();

            let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
            let scale = dot(&re, &re0) / dot(&re0, &re0);
            let n_f = n as f64;
            assert!(
                (scale - 1.0).abs() < 1e-9 || (scale / n_f - 1.0).abs() < 1e-9,
                "k={k}: unexpected round-trip scale {scale}"
            );

            let want_r: Vec<f64> = re0.iter().map(|x| x * scale).collect();
            let want_i: Vec<f64> = im0.iter().map(|x| x * scale).collect();
            let d = diff(&re, &im, &want_r, &want_i);
            assert!(
                d.rel_l2 <= 2.0 * tol(k),
                "k={k}: round-trip error {:.3e}",
                d.rel_l2
            );
        }
    }

    // ------------------------------------------------------------- benchmark

    /// Port of the C++ benchmark: lengths 2^1..2^25, error against FFTW and timings.
    /// Run with: cargo test --release bench_vs_fftw -- --ignored --nocapture
    #[test]
    #[ignore = "benchmark: cargo test --release bench_vs_fftw -- --ignored --nocapture"]
    fn bench_vs_fftw() {
        println!(
            "{:>3} {:>9} {:>11} {:>10} {:>11} {:>11} {:>11} {:>11} {:>13} {:>8}",
            "k",
            "N",
            "max|diff|",
            "rel L2",
            "FFTW plan",
            "FFTW exec",
            "sleek cold",
            "sleek warm",
            "warm+bitrev",
            "speedup"
        );

        for k in 1..=25u32 {
            let n = 1usize << k;
            let reps = ((1usize << 22) >> k).clamp(3, 200);
            let (re, im) = random_signal(k, 42);

            // FFTW: planning is timed separately from execution.
            let t = Instant::now();
            let mut plan = fftw_plan(n);
            let fftw_plan_time = t.elapsed().as_secs_f64();

            let mut input: AlignedVec<c64> = AlignedVec::new(n);
            let mut output: AlignedVec<c64> = AlignedVec::new(n);
            let fftw_exec = best_of(reps, || {
                for i in 0..n {
                    input[i] = c64::new(re[i], im[i]);
                }
                let t = Instant::now();
                plan.c2c(&mut input, &mut output).unwrap();
                t.elapsed()
            });
            let (er, ei) = split(&output);

            // sleekfft cold: fresh object, first call, includes any twiddle setup
            // and the bit-reversal (what the C++ benchmark measures).
            let (mut r, mut i) = (re.clone(), im.clone());
            let t = Instant::now();
            let mut fft = Fft::new().unwrap();
            fft.fft(&mut r, &mut i, k).unwrap();
            bit_reverse_permute(&mut r, &mut i, k).unwrap();
            let sleek_cold = t.elapsed().as_secs_f64();

            // sleekfft warm: tables already built.
            let sleek_warm = best_of(reps, || {
                r.copy_from_slice(&re);
                i.copy_from_slice(&im);
                let t = Instant::now();
                fft.fft(&mut r, &mut i, k).unwrap();
                t.elapsed()
            });
            let sleek_warm_br = best_of(reps, || {
                r.copy_from_slice(&re);
                i.copy_from_slice(&im);
                let t = Instant::now();
                fft.fft(&mut r, &mut i, k).unwrap();
                bit_reverse_permute(&mut r, &mut i, k).unwrap();
                t.elapsed()
            });

            // The last run above left a natural-order result in (r, i).
            let d = diff(&r, &i, &er, &ei);
            assert!(
                d.rel_l2 <= tol(k),
                "k={k}: relative L2 error {:.3e}",
                d.rel_l2
            );

            println!(
                "{:>3} {:>9} {:>11.3e} {:>10.2e} {:>11.7} {:>11.7} {:>11.7} {:>11.7} {:>13.7} {:>7.2}x",
                k,
                n,
                d.max_abs,
                d.rel_l2,
                fftw_plan_time,
                fftw_exec,
                sleek_cold,
                sleek_warm,
                sleek_warm_br,
                fftw_exec / sleek_warm_br
            );
        }
    }
}
