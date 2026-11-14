#define _POSIX_C_SOURCE 199309L
#include <fftw3.h>
#include <sleekfft/fft.h>
#include <stdint.h>
#include <stdlib.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

fftw_complex a[1 << 25];
double ar[1 << 25], ai[1 << 25];

static inline int64_t now_ns(void) {
#if defined(_WIN32)
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (int64_t)(1e9 * c.QuadPart / f.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
#endif
}

void test(const int N) {
    const int k = __builtin_ctz(N);

    for (int i = 0; i < N; ++i) {
        a[i][0] = ar[i] = rand();
        a[i][1] = ai[i] = rand();
    }

    const int64_t t1 = now_ns();

    const fftw_plan p = fftw_plan_dft_1d(N, a, a, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    const int64_t t2 = now_ns();

    sleekfft_context* c = sleekfft_create_context();
    sleekfft_forward_transform(c, ar, ai, k);
    sleekfft_destroy_context(c);
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            f64 tmp = ar[i];
            ar[i] = ar[j];
            ar[j] = tmp;
            tmp = ai[i];
            ai[i] = ai[j];
            ai[j] = tmp;
        }
    }

    const int64_t t3 = now_ns();

    double max_diff = 0;
    for (int i = 0; i < N; ++i) {
        f64 d1 = fabs(a[i][0] - ar[i]);
        f64 d2 = fabs(a[i][1] - ai[i]);
        max_diff = fmax(max_diff, fmax(d1, d2));
    }

    printf("%-20s: %d\n", "Length", N);
    printf("%-20s: %.7e\n", "Max Difference", max_diff);
    printf("%-20s: %.7f\n", "Time Taken FFTW", (t2 - t1) / 1e9);
    printf("%-20s: %.7f\n", "Time Taken SleekFFT", (t3 - t2) / 1e9);
    printf("\n");
}

int main(void) {
    for (int i = 1; i <= 25; ++i)
        test(1 << i);
    return 0;
}