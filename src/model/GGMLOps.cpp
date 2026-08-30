#include "GGMLOps.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

void GGMLGemmVec(const float *W, const float *x, float *out, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float sum = 0.0f;
        const float *row = W + static_cast<std::size_t>(i) * cols;
        for (int j = 0; j < cols; ++j)
            sum += row[j] * x[j];
        out[i] = sum;
    }
}

void GGMLGemm(const float *A, const float *B, float *C, int m, int n, int k, bool accumulate) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int t = 0; t < k; ++t)
                sum +=
                    A[static_cast<std::size_t>(i) * k + t] * B[static_cast<std::size_t>(t) * n + j];
            float &dst = C[static_cast<std::size_t>(i) * n + j];
            dst = accumulate ? dst + sum : sum;
        }
    }
}

void GGMLSoftmax(float *x, int n) {
    if (n <= 0)
        return;
    const float max = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < n; ++i)
        x[i] /= sum;
}

void GGMLSoftmaxMasked(float *x, const float *mask, int n) {
    if (n <= 0)
        return;
    float max = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i)
        if (mask[i] != -std::numeric_limits<float>::infinity())
            max = std::max(max, x[i]);
    if (!std::isfinite(max))
        max = 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (mask[i] == -std::numeric_limits<float>::infinity()) {
            x[i] = 0.0f;
        } else {
            x[i] = std::exp(x[i] - max);
            sum += x[i];
        }
    }
    for (int i = 0; i < n; ++i)
        x[i] /= sum;
}
