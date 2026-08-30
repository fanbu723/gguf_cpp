#include "GGMLRope.hpp"

#include <cmath>

void GGMLRopeNeox(const float *x, float *y, int head_dim, int n_rot, int pos, float freq_base) {
    // 未旋转部分原样复制
    for (int i = 0; i < head_dim; ++i)
        y[i] = x[i];

    const float theta_scale = std::pow(freq_base, -2.0f / static_cast<float>(n_rot));
    const int half = n_rot / 2;
    for (int i = 0; i < half; ++i) {
        const float theta = static_cast<float>(pos) * std::pow(theta_scale, static_cast<float>(i));
        const float c = std::cos(theta);
        const float s = std::sin(theta);
        const float x0 = x[i];
        const float x1 = x[i + half];
        y[i] = x0 * c - x1 * s;
        y[i + half] = x0 * s + x1 * c;
    }
}
