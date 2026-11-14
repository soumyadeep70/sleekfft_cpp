#ifndef SLEEKFFT_X64_AVX_FMA_H
#define SLEEKFFT_X64_AVX_FMA_H

#if defined(__x86_64__) || defined(_M_X64)

#ifdef __cplusplus
extern "C" {
#endif

#include "common.h"
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>

typedef f64 __attribute__((vector_size(32))) f64x4;

[[gnu::target("avx,fma")]]
void x64_avx_fma_reserve_twiddles(twiddle_table* table, const size_t k) {
    const size_t m = 1ULL << (k - 1);
    if (table->size >= m)
        return;
    table->size = m;
    free(table->wr);
    free(table->wi);
    table->wr = (f64*)aligned_alloc(32, m <= 4 ? 32UL : m * sizeof(f64));
    table->wi = (f64*)aligned_alloc(32, m <= 4 ? 32UL : m * sizeof(f64));
    const f64 arg = -PI / (f64)m;
    table->wr[0] = 1.0, table->wi[0] = 0.0;
    for (size_t i = 1, j = m >> 1; j; i <<= 1, j >>= 1) {
        const f64 theta = arg * (f64)j;
        table->wr[i] = cos(theta);
        table->wi[i] = sin(theta);
    }
    for (size_t i = 1; i < m; ++i) {
        const size_t p = i & (i - 1), q = 1ULL << __builtin_ctzll(i);
        table->wr[i] = fma(table->wr[p], table->wr[q], -table->wi[p] * table->wi[q]);
        table->wi[i] = fma(table->wr[p], table->wi[q], table->wi[p] * table->wr[q]);
    }
}

[[gnu::target("avx,fma")]]
void x64_avx_fma_forward_transform(const twiddle_table* table, f64* xr, f64* xi, const size_t k) {
    if (k == 1) {
        const f64 ur = xr[0], vr = xr[1];
        const f64 ui = xi[0], vi = xi[1];
        xr[0] = ur + vr, xr[1] = ur - vr;
        xi[0] = ui + vi, xi[1] = ui - vi;
        return;
    }
    if (k & 1) {
        for (size_t i0 = 0, i1 = 1ULL << (k - 1), ie = i1; i0 < ie; i0 += 4, i1 += 4) {
            const f64x4 ur = _mm256_loadu_pd(xr + i0), vr = _mm256_loadu_pd(xr + i1);
            const f64x4 ui = _mm256_loadu_pd(xi + i0), vi = _mm256_loadu_pd(xi + i1);

            _mm256_storeu_pd(xr + i0, ur + vr);
            _mm256_storeu_pd(xi + i0, ui + vi);
            _mm256_storeu_pd(xr + i1, ur - vr);
            _mm256_storeu_pd(xi + i1, ui - vi);
        }
    }
    for (size_t u = (k & 1) + 1, v = 1ULL << ((k & ~1ULL) - 2); v >= 4; u <<= 2, v >>= 2) {
        for (size_t i0 = 0, i1 = v, i2 = i1 + v, i3 = i2 + v; i0 < v;
             i0 += 4, i1 += 4, i2 += 4, i3 += 4) {
            const f64x4 x0r = _mm256_loadu_pd(xr + i0), x0i = _mm256_loadu_pd(xi + i0);
            const f64x4 x1r = _mm256_loadu_pd(xr + i1), x1i = _mm256_loadu_pd(xi + i1);
            const f64x4 x2r = _mm256_loadu_pd(xr + i2), x2i = _mm256_loadu_pd(xi + i2);
            const f64x4 x3r = _mm256_loadu_pd(xr + i3), x3i = _mm256_loadu_pd(xi + i3);

            const f64x4 y0r = x0r + x2r, y0i = x0i + x2i;
            const f64x4 y1r = x1r + x3r, y1i = x1i + x3i;
            const f64x4 y2r = x0r - x2r, y2i = x0i - x2i;
            const f64x4 y3r = x1i - x3i, y3i = x3r - x1r;

            _mm256_storeu_pd(xr + i0, y0r + y1r);
            _mm256_storeu_pd(xi + i0, y0i + y1i);
            _mm256_storeu_pd(xr + i1, y0r - y1r);
            _mm256_storeu_pd(xi + i1, y0i - y1i);
            _mm256_storeu_pd(xr + i2, y2r + y3r);
            _mm256_storeu_pd(xi + i2, y2i + y3i);
            _mm256_storeu_pd(xr + i3, y2r - y3r);
            _mm256_storeu_pd(xi + i3, y2i - y3i);
        }
        for (size_t h = 1; h < u; ++h) {
            const f64x4 w1r = _mm256_broadcast_sd(table->wr + (h << 1)),
                        w1i = _mm256_broadcast_sd(table->wi + (h << 1));
            const f64x4 w2r = _mm256_broadcast_sd(table->wr + h),
                        w2i = _mm256_broadcast_sd(table->wi + h);
            const f64x4 w3r = _mm256_fmsub_pd(w1r, w2r, w1i * w2i),
                        w3i = _mm256_fmadd_pd(w1r, w2i, w1i * w2r);

            for (size_t i0 = h * 4 * v, i1 = i0 + v, i2 = i1 + v, i3 = i2 + v, ie = i1; i0 < ie;
                 i0 += 4, i1 += 4, i2 += 4, i3 += 4) {
                const f64x4 t0r = _mm256_loadu_pd(xr + i0), t0i = _mm256_loadu_pd(xi + i0);
                const f64x4 t1r = _mm256_loadu_pd(xr + i1), t1i = _mm256_loadu_pd(xi + i1);
                const f64x4 t2r = _mm256_loadu_pd(xr + i2), t2i = _mm256_loadu_pd(xi + i2);
                const f64x4 t3r = _mm256_loadu_pd(xr + i3), t3i = _mm256_loadu_pd(xi + i3);

                const f64x4 x0r = t0r, x0i = t0i;
                const f64x4 x1r = _mm256_fmsub_pd(t1r, w1r, t1i * w1i),
                            x1i = _mm256_fmadd_pd(t1r, w1i, t1i * w1r);
                const f64x4 x2r = _mm256_fmsub_pd(t2r, w2r, t2i * w2i),
                            x2i = _mm256_fmadd_pd(t2r, w2i, t2i * w2r);
                const f64x4 x3r = _mm256_fmsub_pd(t3r, w3r, t3i * w3i),
                            x3i = _mm256_fmadd_pd(t3r, w3i, t3i * w3r);

                const f64x4 y0r = x0r + x2r, y0i = x0i + x2i;
                const f64x4 y1r = x1r + x3r, y1i = x1i + x3i;
                const f64x4 y2r = x0r - x2r, y2i = x0i - x2i;
                const f64x4 y3r = x1i - x3i, y3i = x3r - x1r;

                _mm256_storeu_pd(xr + i0, y0r + y1r);
                _mm256_storeu_pd(xi + i0, y0i + y1i);
                _mm256_storeu_pd(xr + i1, y0r - y1r);
                _mm256_storeu_pd(xi + i1, y0i - y1i);
                _mm256_storeu_pd(xr + i2, y2r + y3r);
                _mm256_storeu_pd(xi + i2, y2i + y3i);
                _mm256_storeu_pd(xr + i3, y2r - y3r);
                _mm256_storeu_pd(xi + i3, y2i - y3i);
            }
        }
    }
    for (size_t h = 0, u = 1ULL << (k - 2); h < u; ++h) {
        const f64 w1r = table->wr[h << 1], w1i = table->wi[h << 1], w2r = table->wr[h],
                  w2i = table->wi[h];
        const f64 w3r = fma(w1r, w2r, -w1i * w2i), w3i = fma(w1r, w2i, w1i * w2r);

        size_t i0 = h << 2, i1 = i0 + 1, i2 = i1 + 1, i3 = i2 + 1;
        const f64 x0r = xr[i0], x0i = xi[i0];
        const f64 x1r = fma(xr[i1], w1r, -xi[i1] * w1i), x1i = fma(xr[i1], w1i, xi[i1] * w1r);
        const f64 x2r = fma(xr[i2], w2r, -xi[i2] * w2i), x2i = fma(xr[i2], w2i, xi[i2] * w2r);
        const f64 x3r = fma(xr[i3], w3r, -xi[i3] * w3i), x3i = fma(xr[i3], w3i, xi[i3] * w3r);

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

[[gnu::target("avx,fma")]]
void x64_avx_fma_inverse_transform(const twiddle_table* table, f64* xr, f64* xi, const size_t k) {
    if (k == 1) {
        const f64 ur = xr[0], vr = xr[1];
        const f64 ui = xi[0], vi = xi[1];
        xr[0] = ur + vr, xr[1] = ur - vr;
        xi[0] = ui + vi, xi[1] = ui - vi;
        return;
    }
    for (size_t h = 0, u = 1ULL << (k - 2); h < u; ++h) {
        const f64 w1r = table->wr[h << 1], w1i = -table->wi[h << 1], w2r = table->wr[h],
                  w2i = -table->wi[h];
        const f64 w3r = fma(w1r, w2r, -w1i * w2i), w3i = fma(w1r, w2i, w1i * w2r);

        size_t i0 = h << 2, i1 = i0 + 1, i2 = i1 + 1, i3 = i2 + 1;
        const f64 x0r = xr[i0] + xr[i1], x0i = xi[i0] + xi[i1];
        const f64 x1r = xr[i0] - xr[i1], x1i = xi[i0] - xi[i1];
        const f64 x2r = xr[i2] + xr[i3], x2i = xi[i2] + xi[i3];
        const f64 x3r = xi[i3] - xi[i2], x3i = xr[i2] - xr[i3];

        const f64 y0r = x0r + x2r, y0i = x0i + x2i;
        const f64 y1r = x1r + x3r, y1i = x1i + x3i;
        const f64 y2r = x0r - x2r, y2i = x0i - x2i;
        const f64 y3r = x1r - x3r, y3i = x1i - x3i;

        xr[i0] = y0r, xi[i0] = y0i;
        xr[i1] = fma(y1r, w1r, -y1i * w1i), xi[i1] = fma(y1r, w1i, y1i * w1r);
        xr[i2] = fma(y2r, w2r, -y2i * w2i), xi[i2] = fma(y2r, w2i, y2i * w2r);
        xr[i3] = fma(y3r, w3r, -y3i * w3i), xi[i3] = fma(y3r, w3i, y3i * w3r);
    }
    for (size_t u = k < 4 ? 0 : 1ULL << (k - 4), v = 4; u; u >>= 2, v <<= 2) {
        for (size_t i0 = 0, i1 = v, i2 = i1 + v, i3 = i2 + v; i0 < v;
             i0 += 4, i1 += 4, i2 += 4, i3 += 4) {
            const f64x4 x0r = _mm256_loadu_pd(xr + i0), x0i = _mm256_loadu_pd(xi + i0);
            const f64x4 x1r = _mm256_loadu_pd(xr + i1), x1i = _mm256_loadu_pd(xi + i1);
            const f64x4 x2r = _mm256_loadu_pd(xr + i2), x2i = _mm256_loadu_pd(xi + i2);
            const f64x4 x3r = _mm256_loadu_pd(xr + i3), x3i = _mm256_loadu_pd(xi + i3);

            const f64x4 y0r = x0r + x1r, y0i = x0i + x1i;
            const f64x4 y1r = x0r - x1r, y1i = x0i - x1i;
            const f64x4 y2r = x2r + x3r, y2i = x2i + x3i;
            const f64x4 y3r = x3i - x2i, y3i = x2r - x3r;

            _mm256_storeu_pd(xr + i0, y0r + y2r);
            _mm256_storeu_pd(xi + i0, y0i + y2i);
            _mm256_storeu_pd(xr + i1, y1r + y3r);
            _mm256_storeu_pd(xi + i1, y1i + y3i);
            _mm256_storeu_pd(xr + i2, y0r - y2r);
            _mm256_storeu_pd(xi + i2, y0i - y2i);
            _mm256_storeu_pd(xr + i3, y1r - y3r);
            _mm256_storeu_pd(xi + i3, y1i - y3i);
        }
        for (size_t h = 1; h < u; ++h) {
            const f64x4 w1r = _mm256_broadcast_sd(table->wr + (h << 1)),
                        w1i = -_mm256_broadcast_sd(table->wi + (h << 1));
            const f64x4 w2r = _mm256_broadcast_sd(table->wr + h),
                        w2i = -_mm256_broadcast_sd(table->wi + h);
            const f64x4 w3r = _mm256_fmsub_pd(w1r, w2r, w1i * w2i),
                        w3i = _mm256_fmadd_pd(w1r, w2i, w1i * w2r);

            for (size_t i0 = h * 4 * v, i1 = i0 + v, i2 = i1 + v, i3 = i2 + v, ie = i1; i0 < ie;
                 i0 += 4, i1 += 4, i2 += 4, i3 += 4) {
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

                _mm256_storeu_pd(xr + i0, z0r);
                _mm256_storeu_pd(xi + i0, z0i);
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
        for (size_t i0 = 0, i1 = 1ULL << (k - 1), ie = i1; i0 < ie; i0 += 4, i1 += 4) {
            const f64x4 ur = _mm256_loadu_pd(xr + i0), vr = _mm256_loadu_pd(xr + i1);
            const f64x4 ui = _mm256_loadu_pd(xi + i0), vi = _mm256_loadu_pd(xi + i1);

            _mm256_storeu_pd(xr + i0, ur + vr);
            _mm256_storeu_pd(xi + i0, ui + vi);
            _mm256_storeu_pd(xr + i1, ur - vr);
            _mm256_storeu_pd(xi + i1, ui - vi);
        }
    }
}

[[gnu::target("avx,fma")]]
void x64_avx_fma_cyclic_conv(const twiddle_table* table, f64* xr, f64* xi, const size_t k) {
    const size_t m = 1ULL << k;
    x64_avx_fma_forward_transform(table, xr, xi, k);
    xi[0] = 4 * xr[0] * xi[0];
    xi[1] = 4 * xr[1] * xi[1];
    xr[0] = 0, xr[1] = 0;
    for (size_t i = 2; i < m; i <<= 1) {
        for (size_t p = i, pe = i << 1; p < pe; p += 2) {
            const size_t q = p ^ (i - 1);
            const f64 ur = xr[p] - xr[q], ui = xi[p] + xi[q];
            const f64 vr = xr[p] + xr[q], vi = xi[p] - xi[q];
            xr[p] = fma(ur, vr, -ui * vi);
            xi[p] = fma(ur, vi, ui * vr);
            xr[q] = -xr[p];
            xi[q] = xi[p];
        }
    }
    const f64 fc = 0.25 / (f64)m;
    for (size_t i = 0, j = 0; i < m; i += 2, ++j) {
        const f64 ar = xr[i] + xr[i | 1], ai = xi[i] + xi[i | 1];
        const f64 br = xr[i] - xr[i | 1], bi = xi[i] - xi[i | 1];
        const f64 cr = fma(br, table->wr[j], bi * table->wi[j]);
        const f64 ci = fma(bi, table->wr[j], -br * table->wi[j]);
        xr[j] = (cr + ai) * fc;
        xi[j] = (ci - ar) * fc;
    }
    x64_avx_fma_inverse_transform(table, xr, xi, k - 1);
}

#ifdef __cplusplus
}
#endif

#endif // defined(__x86_64__) || defined(_M_X64)

#endif // SLEEKFFT_X64_AVX_FMA_H
