/**
 * Author: Soumyadeep Dash
 * Last Edited: 10/08/24
 */

#pragma once

#include <cmath>
#include <immintrin.h>

namespace FFT_AVX_FMA {
    using usize = unsigned long long;
    using f64 = double;
    using f64x4 = f64 __attribute__((vector_size(32)));

    constexpr f64 PI = 3.141592653589793238462;

    [[gnu::target("avx", "fma")]]
    inline void reserve(f64 *&wr, f64 *&wi, usize &size, const usize k) {
        const usize m = 1ULL << (k - 1);
        if (size >= m) return;
        size = m;
        wr = static_cast<f64*>(_mm_malloc(m * sizeof(f64), 32));
        wi = static_cast<f64*>(_mm_malloc(m * sizeof(f64), 32));
        const f64 arg = -PI / static_cast<f64>(m);
        wr[0] = 1.0, wi[0] = 0.0;
        for (usize i = 1, j = m >> 1; j; i <<= 1, j >>= 1) {
            const f64 theta = arg * static_cast<f64>(j);
            wr[i] = std::cos(theta);
            wi[i] = std::sin(theta);
        }
        for (usize i = 1; i < m; ++i) {
            const usize p = i & (i - 1), q = 1ULL << __builtin_ctzll(i);
            wr[i] = std::fma(wr[p], wr[q], -wi[p] * wi[q]);
            wi[i] = std::fma(wr[p], wi[q], wi[p] * wr[q]);
        }
    }

    [[gnu::target("avx", "fma")]]
    inline void fft(const f64 *wr, const f64* wi, f64 *xr, f64 *xi, const usize k) {
        if (k == 1) {
            const f64 ur = xr[0], vr = xr[1];
            const f64 ui = xi[0], vi = xi[1];
            xr[0] = ur + vr, xr[1] = ur - vr;
            xi[0] = ui + vi, xi[1] = ui - vi;
            return;
        }
        if (k & 1) {
            for (usize i0 = 0, i1 = 1ULL << (k - 1), ie = i1; i0 < ie; i0 += 4, i1 += 4) {
                const f64x4 ur = _mm256_loadu_pd(xr + i0), vr = _mm256_loadu_pd(xr + i1);
                const f64x4 ui = _mm256_loadu_pd(xi + i0), vi = _mm256_loadu_pd(xi + i1);

                _mm256_storeu_pd(xr + i0, ur + vr); _mm256_storeu_pd(xi + i0, ui + vi);
                _mm256_storeu_pd(xr + i1, ur - vr); _mm256_storeu_pd(xi + i1, ui - vi);
            }
        }
        for (usize u = (k & 1) + 1, v = 1ULL << ((k & ~1ULL) - 2); v >= 4; u <<= 2, v >>= 2) {
            for (usize i0 = 0, i1 = v, i2 = i1 + v, i3 = i2 + v; i0 < v; i0 += 4, i1 += 4, i2 += 4, i3 += 4) {
                const f64x4 x0r = _mm256_loadu_pd(xr + i0), x0i = _mm256_loadu_pd(xi + i0);
                const f64x4 x1r = _mm256_loadu_pd(xr + i1), x1i = _mm256_loadu_pd(xi + i1);
                const f64x4 x2r = _mm256_loadu_pd(xr + i2), x2i = _mm256_loadu_pd(xi + i2);
                const f64x4 x3r = _mm256_loadu_pd(xr + i3), x3i = _mm256_loadu_pd(xi + i3);

                const f64x4 y0r = x0r + x2r, y0i = x0i + x2i;
                const f64x4 y1r = x1r + x3r, y1i = x1i + x3i;
                const f64x4 y2r = x0r - x2r, y2i = x0i - x2i;
                const f64x4 y3r = x1i - x3i, y3i = x3r - x1r;

                _mm256_storeu_pd(xr + i0, y0r + y1r); _mm256_storeu_pd(xi + i0, y0i + y1i);
                _mm256_storeu_pd(xr + i1, y0r - y1r); _mm256_storeu_pd(xi + i1, y0i - y1i);
                _mm256_storeu_pd(xr + i2, y2r + y3r); _mm256_storeu_pd(xi + i2, y2i + y3i);
                _mm256_storeu_pd(xr + i3, y2r - y3r); _mm256_storeu_pd(xi + i3, y2i - y3i);
            }
            for (usize h = 1; h < u; ++h) {
                const f64x4 w1r = _mm256_broadcast_sd(wr + (h << 1)), w1i = _mm256_broadcast_sd(wi + (h << 1));
                const f64x4 w2r = _mm256_broadcast_sd(wr + h), w2i = _mm256_broadcast_sd(wi + h);
                const f64x4 w3r = _mm256_fmsub_pd(w1r, w2r, w1i * w2i), w3i = _mm256_fmadd_pd(w1r, w2i, w1i * w2r);

                for (usize i0 = h * 4 * v, i1 = i0 + v, i2 = i1 + v, i3 = i2 + v, ie = i1; i0 < ie; i0 += 4, i1 += 4, i2 += 4, i3 += 4) {
                    const f64x4 t0r = _mm256_loadu_pd(xr + i0), t0i = _mm256_loadu_pd(xi + i0);
                    const f64x4 t1r = _mm256_loadu_pd(xr + i1), t1i = _mm256_loadu_pd(xi + i1);
                    const f64x4 t2r = _mm256_loadu_pd(xr + i2), t2i = _mm256_loadu_pd(xi + i2);
                    const f64x4 t3r = _mm256_loadu_pd(xr + i3), t3i = _mm256_loadu_pd(xi + i3);

                    const f64x4 x0r = t0r, x0i = t0i;
                    const f64x4 x1r = _mm256_fmsub_pd(t1r, w1r, t1i * w1i), x1i = _mm256_fmadd_pd(t1r, w1i, t1i * w1r);
                    const f64x4 x2r = _mm256_fmsub_pd(t2r, w2r, t2i * w2i), x2i = _mm256_fmadd_pd(t2r, w2i, t2i * w2r);
                    const f64x4 x3r = _mm256_fmsub_pd(t3r, w3r, t3i * w3i), x3i = _mm256_fmadd_pd(t3r, w3i, t3i * w3r);

                    const f64x4 y0r = x0r + x2r, y0i = x0i + x2i;
                    const f64x4 y1r = x1r + x3r, y1i = x1i + x3i;
                    const f64x4 y2r = x0r - x2r, y2i = x0i - x2i;
                    const f64x4 y3r = x1i - x3i, y3i = x3r - x1r;

                    _mm256_storeu_pd(xr + i0, y0r + y1r); _mm256_storeu_pd(xi + i0, y0i + y1i);
                    _mm256_storeu_pd(xr + i1, y0r - y1r); _mm256_storeu_pd(xi + i1, y0i - y1i);
                    _mm256_storeu_pd(xr + i2, y2r + y3r); _mm256_storeu_pd(xi + i2, y2i + y3i);
                    _mm256_storeu_pd(xr + i3, y2r - y3r); _mm256_storeu_pd(xi + i3, y2i - y3i);
                }
            }
        }
        for (usize h = 0, u = 1ULL << (k - 2); h < u; ++h) {
            const f64 w1r = wr[h << 1], w1i = wi[h << 1], w2r = wr[h], w2i = wi[h];
            const f64 w3r = std::fma(w1r, w2r, -w1i * w2i), w3i = std::fma(w1r, w2i, w1i * w2r);

            usize i0 = h << 2, i1 = i0 + 1, i2 = i1 + 1, i3 = i2 + 1;
            const f64 x0r = xr[i0], x0i = xi[i0];
            const f64 x1r = std::fma(xr[i1], w1r, -xi[i1] * w1i), x1i = std::fma(xr[i1], w1i, xi[i1] * w1r);
            const f64 x2r = std::fma(xr[i2], w2r, -xi[i2] * w2i), x2i = std::fma(xr[i2], w2i, xi[i2] * w2r);
            const f64 x3r = std::fma(xr[i3], w3r, -xi[i3] * w3i), x3i = std::fma(xr[i3], w3i, xi[i3] * w3r);

            const f64 y0r = x0r + x2r, y0i = x0i + x2i;
            const f64 y1r = x1r + x3r, y1i = x1i + x3i;
            const f64 y2r = x0r - x2r, y2i = x0i - x2i;
            const f64 y3r = x1i - x3i, y3i = x3r - x1r;

            xr[i0] = y0r + y1r, xi[i0] = y0i + y1i;
            xr[i1] = y0r - y1r, xi[i1] = y0i - y1i;
            xr[i2] = y2r + y3r, xi[i2] = y2i + y3i;
            xr[i3] = y2r - y3r, xi[i3] = y2i - y3i;
        }
    }

    [[gnu::target("avx", "fma")]]
    inline void ifft(const f64 *wr, const f64* wi, f64 *xr, f64 *xi, const usize k) {
        if (k == 1) {
            const f64 ur = xr[0], vr = xr[1];
            const f64 ui = xi[0], vi = xi[1];
            xr[0] = ur + vr, xr[1] = ur - vr;
            xi[0] = ui + vi, xi[1] = ui - vi;
            return;
        }
        for (usize h = 0, u = 1ULL << (k - 2); h < u; ++h) {
            const f64 w1r = wr[h << 1], w1i = -wi[h << 1], w2r = wr[h], w2i = -wi[h];
            const f64 w3r = std::fma(w1r, w2r, -w1i * w2i), w3i = std::fma(w1r, w2i, w1i * w2r);

            usize i0 = h << 2, i1 = i0 + 1, i2 = i1 + 1, i3 = i2 + 1;
            const f64 x0r = xr[i0] + xr[i1], x0i = xi[i0] + xi[i1];
            const f64 x1r = xr[i0] - xr[i1], x1i = xi[i0] - xi[i1];
            const f64 x2r = xr[i2] + xr[i3], x2i = xi[i2] + xi[i3];
            const f64 x3r = xi[i3] - xi[i2], x3i = xr[i2] - xr[i3];

            const f64 y0r = x0r + x2r, y0i = x0i + x2i;
            const f64 y1r = x1r + x3r, y1i = x1i + x3i;
            const f64 y2r = x0r - x2r, y2i = x0i - x2i;
            const f64 y3r = x1r - x3r, y3i = x1i - x3i;

            xr[i0] = y0r, xi[i0] = y0i;
            xr[i1] = std::fma(y1r, w1r, -y1i * w1i), xi[i1] = std::fma(y1r, w1i, y1i * w1r);
            xr[i2] = std::fma(y2r, w2r, -y2i * w2i), xi[i2] = std::fma(y2r, w2i, y2i * w2r);
            xr[i3] = std::fma(y3r, w3r, -y3i * w3i), xi[i3] = std::fma(y3r, w3i, y3i * w3r);
        }
        for (usize u = k < 4 ? 0 : 1ULL << (k - 4), v = 4; u; u >>= 2, v <<= 2) {
            for (usize i0 = 0, i1 = v, i2 = i1 + v, i3 = i2 + v; i0 < v; i0 += 4, i1 += 4, i2 += 4, i3 += 4) {
                const f64x4 x0r = _mm256_loadu_pd(xr + i0), x0i = _mm256_loadu_pd(xi + i0);
                const f64x4 x1r = _mm256_loadu_pd(xr + i1), x1i = _mm256_loadu_pd(xi + i1);
                const f64x4 x2r = _mm256_loadu_pd(xr + i2), x2i = _mm256_loadu_pd(xi + i2);
                const f64x4 x3r = _mm256_loadu_pd(xr + i3), x3i = _mm256_loadu_pd(xi + i3);

                const f64x4 y0r = x0r + x1r, y0i = x0i + x1i;
                const f64x4 y1r = x0r - x1r, y1i = x0i - x1i;
                const f64x4 y2r = x2r + x3r, y2i = x2i + x3i;
                const f64x4 y3r = x3i - x2i, y3i = x2r - x3r;

                _mm256_storeu_pd(xr + i0, y0r + y2r); _mm256_storeu_pd(xi + i0, y0i + y2i);
                _mm256_storeu_pd(xr + i1, y1r + y3r); _mm256_storeu_pd(xi + i1, y1i + y3i);
                _mm256_storeu_pd(xr + i2, y0r - y2r); _mm256_storeu_pd(xi + i2, y0i - y2i);
                _mm256_storeu_pd(xr + i3, y1r - y3r); _mm256_storeu_pd(xi + i3, y1i - y3i);
            }
            for (usize h = 1; h < u; ++h) {
                const f64x4 w1r = _mm256_broadcast_sd(wr + (h << 1)), w1i = -_mm256_broadcast_sd(wi + (h << 1));
                const f64x4 w2r = _mm256_broadcast_sd(wr + h), w2i = -_mm256_broadcast_sd(wi + h);
                const f64x4 w3r = _mm256_fmsub_pd(w1r, w2r, w1i * w2i), w3i = _mm256_fmadd_pd(w1r, w2i, w1i * w2r);

                for (usize i0 = h * 4 * v, i1 = i0 + v, i2 = i1 + v, i3 = i2 + v, ie = i1; i0 < ie; i0 += 4, i1 += 4, i2 += 4, i3 += 4) {
                    const f64x4 x0r = _mm256_loadu_pd(xr + i0), x0i = _mm256_loadu_pd(xi + i0);
                    const f64x4 x1r = _mm256_loadu_pd(xr + i1), x1i = _mm256_loadu_pd(xi + i1);
                    const f64x4 x2r = _mm256_loadu_pd(xr + i2), x2i = _mm256_loadu_pd(xi + i2);
                    const f64x4 x3r = _mm256_loadu_pd(xr + i3), x3i = _mm256_loadu_pd(xi + i3);

                    const f64x4 y0r = x0r + x1r, y0i = x0i + x1i;
                    const f64x4 y1r = x0r - x1r, y1i = x0i - x1i;
                    const f64x4 y2r = x2r + x3r, y2i = x2i + x3i;
                    const f64x4 y3r = x3i - x2i, y3i = x2r - x3r;

                    const f64x4 z0r = y0r + y2r, z0i = y0i + y2i;
                    const f64x4 z1r = y1r + y3r, z1i = y1i + y3i;
                    const f64x4 z2r = y0r - y2r, z2i = y0i - y2i;
                    const f64x4 z3r = y1r - y3r, z3i = y1i - y3i;

                    _mm256_storeu_pd(xr + i0, z0r); _mm256_storeu_pd(xi + i0, z0i);
                    _mm256_storeu_pd(xr + i1, _mm256_fmsub_pd(z1r, w1r, z1i * w1i));
                    _mm256_storeu_pd(xi + i1, _mm256_fmadd_pd(z1r, w1i, z1i * w1r));
                    _mm256_storeu_pd(xr + i2, _mm256_fmsub_pd(z2r, w2r, z2i * w2i));
                    _mm256_storeu_pd(xi + i2, _mm256_fmadd_pd(z2r, w2i, z2i * w2r));
                    _mm256_storeu_pd(xr + i3, _mm256_fmsub_pd(z3r, w3r, z3i * w3i));
                    _mm256_storeu_pd(xi + i3, _mm256_fmadd_pd(z3r, w3i, z3i * w3r));
                }
            }
        }
        if (k & 1) {
            for (usize i0 = 0, i1 = 1ULL << (k - 1), ie = i1; i0 < ie; i0 += 4, i1 += 4) {
                const f64x4 ur = _mm256_loadu_pd(xr + i0), vr = _mm256_loadu_pd(xr + i1);
                const f64x4 ui = _mm256_loadu_pd(xi + i0), vi = _mm256_loadu_pd(xi + i1);

                _mm256_storeu_pd(xr + i0, ur + vr); _mm256_storeu_pd(xi + i0, ui + vi);
                _mm256_storeu_pd(xr + i1, ur - vr); _mm256_storeu_pd(xi + i1, ui - vi);
            }
        }
    }

    [[gnu::target("avx", "fma")]]
    inline void cyclic_conv(const f64 *wr, const f64* wi, f64 *xr, f64 *xi, const usize k) {
        const usize m = 1ULL << k;
        fft(wr, wi, xr, xi, k);
        xi[0] = 4 * xr[0] * xi[0];
        xi[1] = 4 * xr[1] * xi[1];
        xr[0] = 0, xr[1] = 0;
        for (usize i = 2; i < m; i <<= 1) {
            for (usize p = i, pe = i << 1; p < pe; p += 2) {
                const usize q = p xor (i - 1);
                const f64 ur = xr[p] - xr[q], ui = xi[p] + xi[q];
                const f64 vr = xr[p] + xr[q], vi = xi[p] - xi[q];
                xr[p] = std::fma(ur, vr, -ui * vi);
                xi[p] = std::fma(ur, vi, ui * vr);
                xr[q] = -xr[p];
                xi[q] = xi[p];
            }
        }
        const f64 fc = 0.25 / static_cast<f64>(m);
        for (usize i = 0, j = 0; i < m; i += 2, ++j) {
            const f64 ar = xr[i] + xr[i|1], ai = xi[i] + xi[i|1];
            const f64 br = xr[i] - xr[i|1], bi = xi[i] - xi[i|1];
            const f64 cr = std::fma(br, wr[j], bi * wi[j]);
            const f64 ci = std::fma(bi, wr[j], -br * wi[j]);
            xr[j] = (cr + ai) * fc;
            xi[j] = (ci - ar) * fc;
        }
        ifft(wr, wi, xr, xi, k - 1);
    }

};