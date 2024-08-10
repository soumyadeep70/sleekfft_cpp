/**
 * Author: Soumyadeep Dash
 * Last Edited: 08/08/24
 */

#pragma once

#include <cmath>

namespace  FFT_GENERIC {
    using usize = unsigned long long;
    using f64 = double;

    constexpr f64 PI = 3.141592653589793238462;

    inline void reserve(f64 *&wr, f64 *&wi, usize &size, const usize k) {
        const usize m = 1ULL << (k - 1);
        if (size >= m) return;
        size = m;
        wr = static_cast<f64*>(std::malloc(m * sizeof(f64)));
        wi = static_cast<f64*>(std::malloc(m * sizeof(f64)));
        const f64 arg = -PI / static_cast<f64>(m);
        wr[0] = 1.0, wi[0] = 0.0;
        for (usize i = 1, j = m >> 1; j; i <<= 1, j >>= 1) {
            const f64 theta = arg * static_cast<f64>(j);
            wr[i] = std::cos(theta);
            wi[i] = std::sin(theta);
        }
        for (usize i = 1; i < m; ++i) {
            const usize p = i & (i - 1), q = 1ULL << __builtin_ctzll(i);
            wr[i] = wr[p] * wr[q] - wi[p] * wi[q];
            wi[i] = wr[p] * wi[q] + wi[p] * wr[q];
        }
    }

    inline void fft(const f64 *wr, const f64* wi, f64 *xr, f64 *xi, const usize k) {
        if (k == 1) {
            const f64 ur = xr[0], vr = xr[1];
            const f64 ui = xi[0], vi = xi[1];
            xr[0] = ur + vr, xr[1] = ur - vr;
            xi[0] = ui + vi, xi[1] = ui - vi;
            return;
        }
        if (k & 1) {
            for (usize i0 = 0, i1 = 1ULL << (k - 1), ie = i1; i0 < ie; ++i0, ++i1) {
                const f64 ur = xr[i0], vr = xr[i1];
                const f64 ui = xi[i0], vi = xi[i1];
                xr[i0] = ur + vr, xr[i1] = ur - vr;
                xi[i0] = ui + vi, xi[i1] = ui - vi;
            }
        }
        for (usize u = (k & 1) + 1, v = 1ULL << ((k & ~1) - 2); v; u <<= 2, v >>= 2) {
            for (usize i0 = 0, i1 = v, i2 = i1 + v, i3 = i2 + v; i0 < v; ++i0, ++i1, ++i2, ++i3) {
                const f64 x0r = xr[i0] + xr[i2], x0i = xi[i0] + xi[i2];
                const f64 x1r = xr[i1] + xr[i3], x1i = xi[i1] + xi[i3];
                const f64 x2r = xr[i0] - xr[i2], x2i = xi[i0] - xi[i2];
                const f64 x3r = xi[i1] - xi[i3], x3i = xr[i3] - xr[i1];

                xr[i0] = x0r + x1r, xi[i0] = x0i + x1i;
                xr[i1] = x0r - x1r, xi[i1] = x0i - x1i;
                xr[i2] = x2r + x3r, xi[i2] = x2i + x3i;
                xr[i3] = x2r - x3r, xi[i3] = x2i - x3i;
            }
            for (usize h = 1; h < u; ++h) {
                const f64 w1r = wr[h << 1], w1i = wi[h << 1], w2r = wr[h], w2i = wi[h];
                const f64 w3r = w1r * w2r - w1i * w2i, w3i = w1r * w2i + w1i * w2r;

                for (usize i0 = h * 4 * v, i1 = i0 + v, i2 = i1 + v, i3 = i2 + v, ie = i1; i0 < ie; ++i0, ++i1, ++i2, ++i3) {
                    const f64 t0r = xr[i0], t0i = xi[i0];
                    const f64 t1r = xr[i1] * w1r - xi[i1] * w1i, t1i = xr[i1] * w1i + xi[i1] * w1r;
                    const f64 t2r = xr[i2] * w2r - xi[i2] * w2i, t2i = xr[i2] * w2i + xi[i2] * w2r;
                    const f64 t3r = xr[i3] * w3r - xi[i3] * w3i, t3i = xr[i3] * w3i + xi[i3] * w3r;

                    const f64 x0r = t0r + t2r, x0i = t0i + t2i;
                    const f64 x1r = t1r + t3r, x1i = t1i + t3i;
                    const f64 x2r = t0r - t2r, x2i = t0i - t2i;
                    const f64 x3r = t1i - t3i, x3i = t3r - t1r;

                    xr[i0] = x0r + x1r, xi[i0] = x0i + x1i;
                    xr[i1] = x0r - x1r, xi[i1] = x0i - x1i;
                    xr[i2] = x2r + x3r, xi[i2] = x2i + x3i;
                    xr[i3] = x2r - x3r, xi[i3] = x2i - x3i;
                }
            }
        }
    }

    inline void ifft(const f64 *wr, const f64* wi, f64 *xr, f64 *xi, const usize k) {
        if (k == 1) {
            const f64 ur = xr[0], vr = xr[1];
            const f64 ui = xi[0], vi = xi[1];
            xr[0] = ur + vr, xr[1] = ur - vr;
            xi[0] = ui + vi, xi[1] = ui - vi;
            return;
        }
        for (usize u = 1ULL << (k - 2), v = 1; u; u >>= 2, v <<= 2) {
            for (usize i0 = 0, i1 = v, i2 = i1 + v, i3 = i2 + v; i0 < v; ++i0, ++i1, ++i2, ++i3) {
                const f64 x0r = xr[i0] + xr[i1], x0i = xi[i0] + xi[i1];
                const f64 x1r = xr[i0] - xr[i1], x1i = xi[i0] - xi[i1];
                const f64 x2r = xr[i2] + xr[i3], x2i = xi[i2] + xi[i3];
                const f64 x3r = xi[i3] - xi[i2], x3i = xr[i2] - xr[i3];

                xr[i0] = x0r + x2r, xi[i0] = x0i + x2i;
                xr[i1] = x1r + x3r, xi[i1] = x1i + x3i;
                xr[i2] = x0r - x2r, xi[i2] = x0i - x2i;
                xr[i3] = x1r - x3r, xi[i3] = x1i - x3i;
            }
            for (usize h = 1; h < u; ++h) {
                const f64 w1r = wr[h << 1], w1i = -wi[h << 1], w2r = wr[h], w2i = -wi[h];
                const f64 w3r = w1r * w2r - w1i * w2i, w3i = w1r * w2i + w1i * w2r;

                for (usize i0 = h * 4 * v, i1 = i0 + v, i2 = i1 + v, i3 = i2 + v, ie = i1; i0 < ie; ++i0, ++i1, ++i2, ++i3) {
                    const f64 x0r = xr[i0] + xr[i1], x0i = xi[i0] + xi[i1];
                    const f64 x1r = xr[i0] - xr[i1], x1i = xi[i0] - xi[i1];
                    const f64 x2r = xr[i2] + xr[i3], x2i = xi[i2] + xi[i3];
                    const f64 x3r = xi[i3] - xi[i2], x3i = xr[i2] - xr[i3];

                    const f64 y0r = x0r + x2r, y0i = x0i + x2i;
                    const f64 y1r = x1r + x3r, y1i = x1i + x3i;
                    const f64 y2r = x0r - x2r, y2i = x0i - x2i;
                    const f64 y3r = x1r - x3r, y3i = x1i - x3i;

                    xr[i0] = y0r, xi[i0] = y0i;
                    xr[i1] = y1r * w1r - y1i * w1i, xi[i1] = y1r * w1i + y1i * w1r;
                    xr[i2] = y2r * w2r - y2i * w2i, xi[i2] = y2r * w2i + y2i * w2r;
                    xr[i3] = y3r * w3r - y3i * w3i, xi[i3] = y3r * w3i + y3i * w3r;
                }
            }
        }
        if (k & 1) {
            for (usize i0 = 0, i1 = 1 << (k - 1), ie = i1; i0 < ie; ++i0, ++i1) {
                const f64 ur = xr[i0], vr = xr[i1];
                const f64 ui = xi[i0], vi = xi[i1];
                xr[i0] = ur + vr, xi[i0] = ui + vi;
                xr[i1] = ur - vr, xi[i1] = ui - vi;
            }
        }
    }

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
                xr[p] = ur * vr - ui * vi;
                xi[p] = ur * vi + ui * vr;
                xr[q] = -xr[p];
                xi[q] = xi[p];
            }
        }
        const f64 fc = 0.25 / static_cast<f64>(m);
        for (usize i = 0, j = 0; i < m; i += 2, ++j) {
            const f64 ar = xr[i] + xr[i|1], ai = xi[i] + xi[i|1];
            const f64 br = xr[i] - xr[i|1], bi = xi[i] - xi[i|1];
            const f64 cr = br * wr[j] + bi * wi[j];
            const f64 ci = bi * wr[j] - br * wi[j];
            xr[j] = (cr + ai) * fc;
            xi[j] = (ci - ar) * fc;
        }
        ifft(wr, wi, xr, xi, k - 1);
    }

};