#ifndef SLEEKFFT_COMMON_H
#define SLEEKFFT_COMMON_H

#include <stddef.h>

#define PI 3.141592653589793238462

typedef double f64;

typedef struct twiddle_table {
    f64* wr;
    f64* wi;
    size_t size;
} twiddle_table;

#endif // SLEEKFFT_COMMON_H