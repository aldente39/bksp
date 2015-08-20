#include <complex.h>
#include "bksp_internal.h"

// Horner's scheme
int dhorner (double *gamma, int l, double sigma, double *out) {

    int i, j;
    out[l] = - gamma[l - 1];

    for (i = l - 1; i > 0; i--) {
        out[i] = - sigma * out[i + 1] - gamma[i - 1];
    }

    out[0] = - sigma * out[1] + 1.0;

    for (i = 1; i < l; i++) {
        for (j = l - 1; j >= i; j--) {
            out[j] += - sigma * out[j + 1];
        }
    }

    for (i = 1; i <= l; i++) {
        out[i] = - out[i] / out[0];
    }

    return 0;
}

int zhorner (double _Complex *gamma, int l,
              double _Complex sigma, double _Complex *out) {

    int i, j;
    out[l] = - gamma[l - 1];

    for (i = l - 1; i > 0; i--) {
        out[i] = - sigma * out[i + 1] - gamma[i - 1];
    }

    out[0] = - sigma * out[1] + 1.0;

    for (i = 1; i < l; i++) {
        for (j = l - 1; j >= i; j--) {
            out[j] += - sigma * out[j + 1];
        }
    }

    for (i = 1; i <= l; i++) {
        out[i] = - out[i] / out[0];
    }

    return 0;
}

// Compute a combination
int comb (int n, int r) {
    if (n < 0 || r < 0 || n < r) {
        return 0;
    }
    if (n == r || !r) {
        return 1;
    }
    return comb(n - 1, r) + comb(n - 1, r - 1);
}
