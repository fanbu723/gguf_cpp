#include "GGMLNorm.hpp"

#include <cmath>

void GGMLRmsNorm(const float *x, const float *gamma, float *y, int n, float eps) {
    // 均方根：rms = sqrt(mean(x²) + eps)
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i)
        sum_sq += x[i] * x[i];
    const float rms = std::sqrt(sum_sq / static_cast<float>(n) + eps);
    const float inv = 1.0f / rms;
    for (int i = 0; i < n; ++i)
        y[i] = x[i] * inv * gamma[i];
}
