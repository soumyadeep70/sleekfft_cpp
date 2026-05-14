#include <chrono>
#include <iostream>
#include <random>
#include <memory>

#include "fft.h"
#include "fftw3.h"

void test(const int N) {
    const int k = __builtin_ctz(N);

    static std::uniform_real_distribution<> dist(1, 10000);
    static std::default_random_engine re(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    std::vector<fftw_complex> a(N);
    std::vector<double> ar(N), ai(N);

    for (int i = 0; i < N; ++i) {
        a[i][0] = ar[i] = dist(re);
        a[i][1] = ai[i] = dist(re);
    }

    const auto t1 = std::chrono::high_resolution_clock::now();

    const auto p = fftw_plan_dft_1d(N, a.data(), a.data(), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    const auto t2 = std::chrono::high_resolution_clock::now();

    FFT fft;
    fft.fft(ar.data(), ai.data(), k);
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(ar[i], ar[j]);
            std::swap(ai[i], ai[j]);
        }
    }

    const auto t3 = std::chrono::high_resolution_clock::now();

    double max_diff = 0;
    for (int i = 0; i < N; ++i) {
        max_diff = std::max(max_diff, std::abs(a[i][0] - ar[i]));
        max_diff = std::max(max_diff, std::abs(a[i][1] - ai[i]));
    }

    printf("%-20s: %d\n", "Length", N);
    printf("%-20s: %.7e\n", "Max Difference", max_diff);
    printf("%-20s: %.7f\n", "Time Taken FFTW", (t2 - t1).count() / 1e9);
    printf("%-20s: %.7f\n", "Time Taken SleekFFT", (t3 - t2).count() / 1e9);
    printf("\n");
}

int main() {
    for (int i = 1; i <= 25; ++i) test(1 << i);
    return 0;
}