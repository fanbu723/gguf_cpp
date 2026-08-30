/*
 * test_ssm.cpp — 阶段⑤ 第 3 步：SSM 混合层（Gated Delta Net）单元测试
 *
 * 验证：
 *  - GGMLDeltaNetStep 的递推数学（手算断言）
 *  - GGMLSSMLayer 完整层：跑通 / 有限 / 确定性 / 状态持久 / conv 历史推进
 */

#include <cmath>
#include <iostream>
#include <vector>

#include "GGMLSSM.hpp"

static int g_fail = 0;
static void check(bool ok, const char *msg) {
    if (ok)
        std::cout << "  ✅ " << msg << std::endl;
    else {
        std::cout << "  ❌ " << msg << std::endl;
        ++g_fail;
    }
}
static bool close(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

static GGUFTensorView make_view(const std::string &name, std::vector<float> &buf,
                                std::initializer_list<std::uint64_t> dims) {
    GGUFTensorView v;
    v.name = name;
    v.dims = dims;
    v.type = 0; // F32
    v.data = reinterpret_cast<const std::uint8_t *>(buf.data());
    return v;
}

int main() {
    std::cout << "=== 阶段⑤ 第3步：SSM 混合层单元测试 ===" << std::endl;

    std::cout << "[1] GGMLDeltaNetStep 手算验证" << std::endl;
    {
        // S=[[1,2],[3,4]]（S[i*2+j]），q=[1,0], k=[0,1], v=[2,3], beta=0.5, phi=0.5, Sv=2
        float S[4] = {1, 2, 3, 4};
        const float q[2] = {1, 0}, k[2] = {0, 1}, v[2] = {2, 3};
        float o[2] = {};
        GGMLDeltaNetStep(q, k, v, 0.5f, 0.5f, S, 2, o);
        // 手算：S'=[[0.5,1.0],[1.75,2.5]]；o = [0.5,1.0]/sqrt(2)
        check(close(S[0], 0.5f) && close(S[1], 1.0f) && close(S[2], 1.75f) && close(S[3], 2.5f),
              "S' 手算一致 [[0.5,1.0],[1.75,2.5]]");
        const float inv = 1.0f / std::sqrt(2.0f);
        check(close(o[0], 0.5f * inv) && close(o[1], 1.0f * inv), "o = S'ᵀq/√2 手算一致");
    }

    std::cout << "[2] GGMLSSMLayer 完整层（假权重）" << std::endl;
    {
        // 小配置：hidden=4, d_state=2, n_group=1, d_inner=2, conv_kernel=2, ffn=4
        GGUFModelConfig cfg;
        cfg.embedding_length = 4;
        cfg.ssm_state_size = 2;
        cfg.ssm_group_count = 1;
        cfg.ssm_inner_size = 2;
        cfg.ssm_conv_kernel = 2;
        cfg.feed_forward_length = 4;
        cfg.rms_eps = 1e-6f;

        // conv_dim = 2*key_dim + value_dim = 2*2 + 2 = 6
        std::vector<float> w_qkv(6 * 4, 0.1f); // [6, hidden=4]
        std::vector<float> w_gate(2 * 4, 0.1f);
        std::vector<float> w_alpha(1 * 4, 0.1f);
        std::vector<float> w_beta(1 * 4, 0.1f);
        std::vector<float> w_out(4 * 2, 0.1f); // [hidden=4, d_inner=2]
        std::vector<float> w_a(1, -0.5f);      // 负衰减参数
        std::vector<float> w_dt(1, 0.0f);
        std::vector<float> w_norm(2, 1.0f);
        std::vector<float> w_conv(6 * 2, 0.1f); // [6 通道 × 2 tap]
        std::vector<float> w_an(4, 1.0f);
        std::vector<float> w_pn(4, 1.0f);
        std::vector<float> w_fg(4 * 4, 0.1f);
        std::vector<float> w_fu(4 * 4, 0.1f);
        std::vector<float> w_fd(4 * 4, 0.1f);

        GGUFTensorView v_qkv = make_view("attn_qkv", w_qkv, {4, 6});
        GGUFTensorView v_gate = make_view("attn_gate", w_gate, {4, 2});
        GGUFTensorView v_alpha = make_view("ssm_alpha", w_alpha, {4, 1});
        GGUFTensorView v_beta = make_view("ssm_beta", w_beta, {4, 1});
        GGUFTensorView v_out = make_view("ssm_out", w_out, {2, 4});
        GGUFTensorView v_a = make_view("ssm_a", w_a, {1});
        GGUFTensorView v_dt = make_view("ssm_dt.bias", w_dt, {1});
        GGUFTensorView v_norm = make_view("ssm_norm", w_norm, {2});
        GGUFTensorView v_conv = make_view("ssm_conv1d", w_conv, {2, 6});
        GGUFTensorView v_an = make_view("attn_norm", w_an, {4});
        GGUFTensorView v_pn = make_view("post_attention_norm", w_pn, {4});
        GGUFTensorView v_fg = make_view("ffn_gate", w_fg, {4, 4});
        GGUFTensorView v_fu = make_view("ffn_up", w_fu, {4, 4});
        GGUFTensorView v_fd = make_view("ffn_down", w_fd, {4, 4});

        GGUFBlockWeights bw;
        bw.attn_qkv = &v_qkv;
        bw.attn_gate = &v_gate;
        bw.ssm_alpha = &v_alpha;
        bw.ssm_beta = &v_beta;
        bw.ssm_out = &v_out;
        bw.ssm_a = &v_a;
        bw.ssm_dt_bias = &v_dt;
        bw.ssm_norm = &v_norm;
        bw.ssm_conv1d = &v_conv;
        bw.attn_norm = &v_an;
        bw.post_attention_norm = &v_pn;
        bw.ffn_gate = &v_fg;
        bw.ffn_up = &v_fu;
        bw.ffn_down = &v_fd;

        const float x[4] = {1, 0, 0, 0};
        float y1[4] = {}, y2[4] = {};

        GGMLSSMState st1;
        st1.init(1, 2, 6);
        GGMLSSMLayer(bw, cfg, st1, x, y1);
        bool fin = true;
        for (float f : y1)
            if (!std::isfinite(f))
                fin = false;
        check(fin, "SSM 层前向输出为有限值");

        GGMLSSMState st2;
        st2.init(1, 2, 6);
        GGMLSSMLayer(bw, cfg, st2, x, y2);
        bool same = true;
        for (int i = 0; i < 4; ++i)
            if (y1[i] != y2[i])
                same = false;
        check(same, "相同输入 → 输出确定性一致");

        // 状态持久：第二次 token 后 S 非零（delta net 更新过）
        GGMLSSMState st3;
        st3.init(1, 2, 6);
        GGMLSSMLayer(bw, cfg, st3, x, y2);
        bool s_nonzero = false;
        for (float s : st3.S)
            if (s != 0.0f)
                s_nonzero = true;
        check(s_nonzero, "递推后状态 S 非零（delta net 生效）");

        // conv 历史推进：第一次调用后 conv_hist[0] = 当前 qkv（非零）
        bool hist_ok = false;
        for (float h : st3.conv_hist)
            if (h != 0.0f)
                hist_ok = true;
        check(hist_ok, "conv 历史已推进（非零）");

        // 状态持久性：同一状态对象连续两个 token，S 继续更新（输出也应变化）
        GGMLSSMState st4;
        st4.init(1, 2, 6);
        float ya[4] = {}, yb[4] = {};
        GGMLSSMLayer(bw, cfg, st4, x, ya);
        const float x2[4] = {0, 1, 0, 0};
        GGMLSSMLayer(bw, cfg, st4, x2, yb);
        bool changed = false;
        for (int i = 0; i < 4; ++i)
            if (ya[i] != yb[i])
                changed = true;
        check(changed, "连续 token 输入不同 → 输出不同（状态累积）");
    }

    std::cout << (g_fail == 0 ? "\n✅ 全部通过" : "\n❌ 有失败") << std::endl;
    return g_fail == 0 ? 0 : 1;
}
