/*
 * test_model_ops.cpp — 阶段⑤ 基础算子单元测试（不依赖真实模型）
 *
 * 验证：矩阵乘(gemv/gemm)、softmax(带掩码)、RMSNorm、SwiGLU FFN、
 *       RoPE(NEOX)、KV cache + GQA 注意力。
 * 全部用手算/性质断言，覆盖数学正确性。
 */

#include <cmath>
#include <iostream>
#include <vector>

#include "GGMLAttention.hpp"
#include "GGMLFFN.hpp"
#include "GGMLNorm.hpp"
#include "GGMLOps.hpp"
#include "GGMLRope.hpp"

static int g_fail = 0;

static void check(bool ok, const char *msg) {
    if (ok)
        std::cout << "  ✅ " << msg << std::endl;
    else {
        std::cout << "  ❌ " << msg << std::endl;
        ++g_fail;
    }
}

// 浮点近似比较
static bool close(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

int main() {
    std::cout << "=== 阶段⑤ 基础算子单元测试 ===" << std::endl;

    std::cout << "[1] 矩阵向量乘 gemv" << std::endl;
    {
        // W = [[1,2],[3,4],[5,6]] (3x2)，x = [1,1]
        const float W[6] = {1, 2, 3, 4, 5, 6};
        const float x[2] = {1, 1};
        float out[3] = {};
        GGMLGemmVec(W, x, out, 3, 2);
        check(close(out[0], 3) && close(out[1], 7) && close(out[2], 11), "gemv: [3,7,11] 正确");
    }

    std::cout << "[2] 矩阵乘 gemm" << std::endl;
    {
        const float A[4] = {1, 2, 3, 4}; // [[1,2],[3,4]]
        const float B[4] = {1, 0, 0, 1}; // 单位阵
        float C[4] = {};
        GGMLGemm(A, B, C, 2, 2, 2); // 覆盖模式
        check(close(C[0], 1) && close(C[1], 2) && close(C[2], 3) && close(C[3], 4),
              "gemm(覆盖): A·I = A");
        for (int i = 0; i < 4; ++i)
            C[i] = 10.0f;
        GGMLGemm(A, B, C, 2, 2, 2, true); // 累加模式
        check(close(C[0], 11) && close(C[3], 14), "gemm(累加): 10 + A = [11,...,14]");
    }

    std::cout << "[3] softmax" << std::endl;
    {
        float x[3] = {1, 2, 3};
        GGMLSoftmax(x, 3);
        check(close(x[0] + x[1] + x[2], 1.0f), "softmax 之和 = 1");
        check(close(x[0], 0.0900f) && close(x[2], 0.6652f), "softmax 值 [0.09,0.245,0.665]");
    }

    std::cout << "[4] 带掩码 softmax" << std::endl;
    {
        float x[3] = {1, 2, 3};
        const float mask[3] = {0.0f, -std::numeric_limits<float>::infinity(), 0.0f};
        GGMLSoftmaxMasked(x, mask, 3);
        check(x[1] == 0.0f, "masked softmax: 掩码位置输出 0");
        check(close(x[0] + x[2], 1.0f), "masked softmax: 其余位置归一化");
    }

    std::cout << "[5] RMSNorm" << std::endl;
    {
        const float x[3] = {1, 2, 3};
        const float gamma[3] = {1, 1, 1};
        float y[3] = {};
        GGMLRmsNorm(x, gamma, y, 3, 1e-6f);
        const float rms = std::sqrt(14.0f / 3.0f); // mean(x²)=14/3
        check(close(y[0], 1.0f / rms) && close(y[2], 3.0f / rms), "RMSNorm: y = x / rms (gamma=1)");
    }

    std::cout << "[6] SwiGLU FFN" << std::endl;
    {
        // gate=up=I(2x2)，down=I(2x2)，x=[2,3] → hidden=silu(x)·x，out=hidden
        const float gate[4] = {1, 0, 0, 1};
        const float up[4] = {1, 0, 0, 1};
        const float down[4] = {1, 0, 0, 1};
        const float x[2] = {2, 3};
        float out[2] = {};
        GGMLSwiGLU(gate, up, down, x, out, 2, 2);
        const float silu2 = 2.0f / (1.0f + std::exp(-2.0f));
        const float silu3 = 3.0f / (1.0f + std::exp(-3.0f));
        check(close(out[0], 2.0f * silu2) && close(out[1], 3.0f * silu3),
              "SwiGLU: out = x ⊙ silu(x)");
    }

    std::cout << "[7] RoPE（NEOX）" << std::endl;
    {
        // pos=0：旋转角为 0，输出应等于输入
        const float x[4] = {1, 0, 0, 1};
        float y[4] = {};
        GGMLRopeNeox(x, y, 4, 4, 0, 10000.0f);
        check(close(y[0], 1) && close(y[1], 0) && close(y[2], 0) && close(y[3], 1),
              "RoPE pos=0 保持不变");

        // pos=1：每个旋转对 (i, i+2) 的范数保持不变（旋转保范）
        GGMLRopeNeox(x, y, 4, 4, 1, 10000.0f);
        const float n0 = x[0] * x[0] + x[2] * x[2];
        const float n1 = y[0] * y[0] + y[2] * y[2];
        const float n2 = x[1] * x[1] + x[3] * x[3];
        const float n3 = y[1] * y[1] + y[3] * y[3];
        check(close(n0, n1, 1e-3f) && close(n2, n3, 1e-3f), "RoPE 旋转保范");

        // n_rot < head_dim：超出 n_rot 的维度原样复制
        const float big[6] = {1, 0, 0, 1, 7, 8};
        float yb[6] = {};
        GGMLRopeNeox(big, yb, 6, 4, 1, 10000.0f);
        check(close(yb[4], 7) && close(yb[5], 8), "RoPE 未旋转部分原样复制");
    }

    std::cout << "[8] KV cache + GQA 注意力" << std::endl;
    {
        GGMLKVCache cache;
        cache.init(1, 2, 2, 4); // 1 个 KV 头，dim_k=2, dim_v=2, 最多 4 位置
        const float k0[2] = {1, 0}, v0[2] = {1, 0};
        const float k1[2] = {0, 1}, v1[2] = {0, 1};
        cache.store(0, 0, k0, v0);
        cache.store(0, 1, k1, v1);
        check(cache.len == 2, "cache.len = 2");
        check(close(cache.k_at(0, 0)[1], 0) && close(cache.v_at(0, 1)[1], 1),
              "cache store/at 存取正确");

        // q = 两个 head：[1,0] 和 [0,1]，scale=1，无分组（2 head 共享 1 kv head）
        const float q[4] = {1, 0, 0, 1};
        float out[4] = {};
        GGMLAttentionGQA(q, cache, 2, 1, 1.0f, out);
        // head0：scores=[1,0]→softmax=[0.7311,0.2689]→out=[0.7311,0.2689]
        // head1：scores=[0,1]→softmax=[0.2689,0.7311]→out=[0.2689,0.7311]
        check(close(out[0], 0.7311f) && close(out[1], 0.2689f), "GQA head0 输出正确");
        check(close(out[2], 0.2689f) && close(out[3], 0.7311f), "GQA head1 输出正确");
    }

    std::cout << (g_fail == 0 ? "\n✅ 全部通过" : "\n❌ 有失败") << std::endl;
    return g_fail == 0 ? 0 : 1;
}
