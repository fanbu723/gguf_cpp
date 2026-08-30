/*
 * GGMLFFN.hpp — SwiGLU 前馈网络（阶段 ⑤）
 *
 * Qwen 的 FFN 用 SwiGLU 门控激活：
 *
 *   hidden[i] = silu(x · gate[i]) ⊙ (x · up[i])
 *   out       = hidden · down
 *
 * 其中 silu(z) = z · sigmoid(z)。gate/up 把输入升维到 hidden，down 降回输入维。
 *
 * 权重布局（GGUF 列主序，等价行主序 [out, in]）：
 *   ffn_gate.weight : [hidden, in]
 *   ffn_up.weight   : [hidden, in]
 *   ffn_down.weight : [in, hidden]
 */

#pragma once

#include <cmath>

/**
 * @brief SwiGLU 前馈网络前向：out = down(silu(x·gateᵀ) ⊙ (x·upᵀ))
 * @param gate 门控权重，行主序 [hidden, in]
 * @param up 升维权重，行主序 [hidden, in]
 * @param down 降维权重，行主序 [in, hidden]
 * @param x 输入向量（in 个元素）
 * @param out 输出向量（in 个元素）
 * @param in 输入维度
 * @param hidden FFN 中间维度（feed_forward_length）
 */
void GGMLSwiGLU(const float *gate, const float *up, const float *down, const float *x, float *out,
                int in, int hidden);

/**
 * @brief SiLU（Sigmoid Linear Unit）激活函数
 * @param z 输入
 * @return silu(z) = z · sigmoid(z)
 */
inline float GGMLSiLU(float z) {
    return z / (1.0f + std::exp(-z));
}
