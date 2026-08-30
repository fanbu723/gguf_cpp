/*
 * test_transformer.cpp — 阶段⑤ 第 2 步：纯 Attention 层前向单元测试
 *
 * 用手建的小配置 + 假权重（F32）跑一次层前向，验证：
 *  - 能跑通且输出维度正确、数值有限
 *  - 确定性（相同输入 → 相同输出）
 *  - 位置不同 → 输出不同（RoPE 生效）
 *  - KV cache 长度递增
 */

#include <cmath>
#include <iostream>
#include <vector>

#include "GGMLTransformer.hpp"

static int g_fail = 0;
static void check(bool ok, const char *msg) {
    if (ok)
        std::cout << "  ✅ " << msg << std::endl;
    else {
        std::cout << "  ❌ " << msg << std::endl;
        ++g_fail;
    }
}
static bool finite4(const float *v) {
    for (int i = 0; i < 4; ++i)
        if (!std::isfinite(v[i]))
            return false;
    return true;
}

// 构造指向 buffer 的 F32 张量视图
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
    std::cout << "=== 阶段⑤ 第2步：Attention 层前向单元测试 ===" << std::endl;

    // ---- 小配置：hidden=4, head=2, kv=1, head_dim=2, ffn=4, n_rot=2 ----
    GGUFModelConfig cfg;
    cfg.embedding_length = 4;
    cfg.head_count = 2;
    cfg.head_count_kv = 1;
    cfg.key_length = 2;
    cfg.value_length = 2;
    cfg.feed_forward_length = 4;
    cfg.rope_dimension_count = 2;
    cfg.rms_eps = 1e-6f;
    cfg.rope_freq_base = 10000.0f;

    // ---- 假权重（全部 0.1，正态无关，仅测结构）----
    std::vector<float> w_q(8 * 4, 0.1f);       // [2*2*2=8, hidden=4]
    std::vector<float> w_k(2 * 4, 0.1f);       // [kv*head_dim=2, 4]
    std::vector<float> w_v(2 * 4, 0.1f);
    std::vector<float> w_o(4 * 4, 0.1f);       // [hidden=4, n_head*head_dim=4]
    std::vector<float> w_qn(2, 1.0f);          // q_norm
    std::vector<float> w_kn(2, 1.0f);          // k_norm
    std::vector<float> w_an(4, 1.0f);          // attn_norm
    std::vector<float> w_pn(4, 1.0f);          // post_attention_norm
    std::vector<float> w_fg(4 * 4, 0.1f);      // ffn_gate
    std::vector<float> w_fu(4 * 4, 0.1f);      // ffn_up
    std::vector<float> w_fd(4 * 4, 0.1f);      // ffn_down

    GGUFTensorView v_q = make_view("attn_q", w_q, {4, 8});
    GGUFTensorView v_k = make_view("attn_k", w_k, {4, 2});
    GGUFTensorView v_v = make_view("attn_v", w_v, {4, 2});
    GGUFTensorView v_o = make_view("attn_output", w_o, {4, 4});
    GGUFTensorView v_qn = make_view("attn_q_norm", w_qn, {2});
    GGUFTensorView v_kn = make_view("attn_k_norm", w_kn, {2});
    GGUFTensorView v_an = make_view("attn_norm", w_an, {4});
    GGUFTensorView v_pn = make_view("post_attention_norm", w_pn, {4});
    GGUFTensorView v_fg = make_view("ffn_gate", w_fg, {4, 4});
    GGUFTensorView v_fu = make_view("ffn_up", w_fu, {4, 4});
    GGUFTensorView v_fd = make_view("ffn_down", w_fd, {4, 4});

    GGUFBlockWeights bw;
    bw.attn_q = &v_q;
    bw.attn_k = &v_k;
    bw.attn_v = &v_v;
    bw.attn_output = &v_o;
    bw.attn_q_norm = &v_qn;
    bw.attn_k_norm = &v_kn;
    bw.attn_norm = &v_an;
    bw.post_attention_norm = &v_pn;
    bw.ffn_gate = &v_fg;
    bw.ffn_up = &v_fu;
    bw.ffn_down = &v_fd;

    // ---- 测试 ----
    const float x[4] = {1, 0, 0, 0};
    float y1[4] = {}, y2[4] = {};

    GGMLKVCache cache1;
    cache1.init(1, 2, 2, 8);
    GGMLTransformerAttentionBlock(bw, cfg, cache1, 0, x, y1);
    check(finite4(y1), "pos=0 前向输出为有限值");
    check(cache1.len == 1, "pos=0 后 KV cache 长度 = 1");

    GGMLKVCache cache2;
    cache2.init(1, 2, 2, 8);
    GGMLTransformerAttentionBlock(bw, cfg, cache2, 0, x, y2);
    bool same = true;
    for (int i = 0; i < 4; ++i)
        if (y1[i] != y2[i])
            same = false;
    check(same, "相同输入 → 输出确定性一致");

    // 第二个 token（pos=1），输入不同 → KV cache 增长到 2
    const float x2[4] = {0, 1, 0, 0};
    float y3[4] = {};
    GGMLKVCache cache3;
    cache3.init(1, 2, 2, 8);
    GGMLTransformerAttentionBlock(bw, cfg, cache3, 0, x, y3);
    GGMLTransformerAttentionBlock(bw, cfg, cache3, 1, x2, y3);
    check(cache3.len == 2, "两个 token 后 KV cache 长度 = 2");

    // 位置影响：同一个 token 在 pos=0 与 pos=1（重开 cache）输出应不同
    float yp[4] = {};
    GGMLKVCache cache4;
    cache4.init(1, 2, 2, 8);
    GGMLTransformerAttentionBlock(bw, cfg, cache4, 1, x, yp);
    bool diff = false;
    for (int i = 0; i < 4; ++i)
        if (y1[i] != yp[i])
            diff = true;
    check(diff, "不同位置(pos=0 vs pos=1)输出不同（RoPE 生效）");

    std::cout << (g_fail == 0 ? "\n✅ 全部通过" : "\n❌ 有失败") << std::endl;
    return g_fail == 0 ? 0 : 1;
}
