#include "GGMLFFN.hpp"

#include <vector>

#include "GGMLOps.hpp"

void GGMLSwiGLU(const float *gate, const float *up, const float *down, const float *x, float *out,
                int in, int hidden) {
    std::vector<float> g(static_cast<std::size_t>(hidden));
    std::vector<float> u(static_cast<std::size_t>(hidden));

    // 1. g = x·gateᵀ，u = x·upᵀ（每行 hidden 个输出，输入维 in）
    GGMLGemmVec(gate, x, g.data(), hidden, in);
    GGMLGemmVec(up, x, u.data(), hidden, in);

    // 2. hidden[i] = silu(g[i]) * u[i]（就地用 g 存结果）
    for (int i = 0; i < hidden; ++i)
        g[static_cast<std::size_t>(i)] =
            GGMLSiLU(g[static_cast<std::size_t>(i)]) * u[static_cast<std::size_t>(i)];

    // 3. out = hidden·downᵀ（输入维 hidden，输出维 in）
    GGMLGemmVec(down, g.data(), out, in, hidden);
}
