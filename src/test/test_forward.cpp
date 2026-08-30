/*
 * test_forward.cpp — 阶段⑤ 第 4 步：全模型前向单元测试
 *
 * 手建一个 2 层假模型（blk.0 SSM + blk.1 Attention）+ token_embd + output_norm，
 * 用 GGUFModelWeights::build + GGMLModelState + GGMLForward 跑完整前向，验证：
 *  - logits 维度 = 词汇表大小
 *  - 输出有限 / 确定性
 *  - 不同 token → 不同 logits
 *  - 位置递增时状态累积 → logits 变化
 */

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "GGMLForward.hpp"

static int g_fail = 0;
static void check(bool ok, const char *msg) {
    if (ok)
        std::cout << "  ✅ " << msg << std::endl;
    else {
        std::cout << "  ❌ " << msg << std::endl;
        ++g_fail;
    }
}

static void add_str(GGUFModel &m, const std::string &key, const std::string &v) {
    m.metadata.push_back({key, GGUFValueType::STRING, v});
}
static void add_u32(GGUFModel &m, const std::string &key, std::uint32_t v) {
    m.metadata.push_back({key, GGUFValueType::UINT32, v});
}
static void add_f32(GGUFModel &m, const std::string &key, float v) {
    m.metadata.push_back({key, GGUFValueType::FLOAT32, v});
}
static void add_tensor(GGUFModel &m, const std::string &name,
                       std::initializer_list<std::uint64_t> dims, std::uint64_t offset) {
    GGUFTensorInfo t;
    t.name = name;
    t.n_dimensions = static_cast<std::uint32_t>(dims.size());
    t.dimensions = dims;
    t.data_type = 0; // F32
    t.offset = offset;
    m.tensors.push_back(t);
}

int main() {
    std::cout << "=== 阶段⑤ 第4步：全模型前向单元测试 ===" << std::endl;

    // ---- 构造假模型：hidden=4, 2 层（blk.0 SSM + blk.1 Attention），vocab=5 ----
    GGUFModel model;
    // 假数据区：权重填有区分度的值（避免全 0.1 退化导致输出对输入/历史不敏感）
    std::vector<float> fake(4096);
    for (std::size_t i = 0; i < fake.size(); ++i)
        fake[i] = 0.1f * static_cast<float>((i % 7) + 1);
    // token_embd 填每列不同：token v 的 embedding = v+1（保证 token 区分）
    for (int v = 0; v < 5; ++v)
        for (int i = 0; i < 4; ++i)
            fake[static_cast<std::size_t>(v * 4 + i)] = static_cast<float>(v + 1);
    model.data.data_ptr = reinterpret_cast<const std::uint8_t *>(fake.data());
    model.data.data_size = fake.size() * sizeof(float);

    add_str(model, "general.architecture", "qwen35");
    add_u32(model, "qwen35.block_count", 2);
    add_u32(model, "qwen35.embedding_length", 4);
    add_u32(model, "qwen35.feed_forward_length", 4);
    add_u32(model, "qwen35.attention.head_count", 2);
    add_u32(model, "qwen35.attention.head_count_kv", 1);
    add_u32(model, "qwen35.attention.key_length", 2);
    add_u32(model, "qwen35.attention.value_length", 2);
    add_f32(model, "qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
    add_u32(model, "qwen35.ssm.conv_kernel", 2);
    add_u32(model, "qwen35.ssm.state_size", 2);
    add_u32(model, "qwen35.ssm.group_count", 1);
    add_u32(model, "qwen35.ssm.time_step_rank", 1);
    add_u32(model, "qwen35.ssm.inner_size", 2);
    add_u32(model, "qwen35.full_attention_interval", 1);
    add_u32(model, "qwen35.rope.dimension_count", 2);
    add_f32(model, "qwen35.rope.freq_base", 10000.0f);

    std::uint64_t off = 0;
    add_tensor(model, "token_embd.weight", {4, 5}, off);
    off += 20;
    add_tensor(model, "output_norm.weight", {4}, off);
    off += 4;
    // blk.0：SSM 层
    add_tensor(model, "blk.0.attn_norm.weight", {4}, off);
    off += 4;
    add_tensor(model, "blk.0.ssm_a", {1}, off);
    off += 1;
    add_tensor(model, "blk.0.ssm_conv1d.weight", {2, 6}, off);
    off += 12;
    add_tensor(model, "blk.0.ssm_dt.bias", {1}, off);
    off += 1;
    add_tensor(model, "blk.0.ssm_alpha.weight", {4, 1}, off);
    off += 4;
    add_tensor(model, "blk.0.ssm_beta.weight", {4, 1}, off);
    off += 4;
    add_tensor(model, "blk.0.attn_qkv.weight", {4, 6}, off);
    off += 24;
    add_tensor(model, "blk.0.attn_gate.weight", {4, 2}, off);
    off += 8;
    add_tensor(model, "blk.0.ssm_norm.weight", {2}, off);
    off += 2;
    add_tensor(model, "blk.0.ssm_out.weight", {2, 4}, off);
    off += 8;
    add_tensor(model, "blk.0.ffn_down.weight", {4, 4}, off);
    off += 16;
    add_tensor(model, "blk.0.ffn_gate.weight", {4, 4}, off);
    off += 16;
    add_tensor(model, "blk.0.ffn_up.weight", {4, 4}, off);
    off += 16;
    add_tensor(model, "blk.0.post_attention_norm.weight", {4}, off);
    off += 4;
    // blk.1：Attention 层
    add_tensor(model, "blk.1.attn_norm.weight", {4}, off);
    off += 4;
    add_tensor(model, "blk.1.attn_q.weight", {4, 8}, off);
    off += 32;
    add_tensor(model, "blk.1.attn_k.weight", {4, 2}, off);
    off += 8;
    add_tensor(model, "blk.1.attn_v.weight", {4, 2}, off);
    off += 8;
    add_tensor(model, "blk.1.attn_output.weight", {4, 4}, off);
    off += 16;
    add_tensor(model, "blk.1.attn_q_norm.weight", {2}, off);
    off += 2;
    add_tensor(model, "blk.1.attn_k_norm.weight", {2}, off);
    off += 2;
    add_tensor(model, "blk.1.ffn_down.weight", {4, 4}, off);
    off += 16;
    add_tensor(model, "blk.1.ffn_gate.weight", {4, 4}, off);
    off += 16;
    add_tensor(model, "blk.1.ffn_up.weight", {4, 4}, off);
    off += 16;
    add_tensor(model, "blk.1.post_attention_norm.weight", {4}, off);
    off += 4;

    // ---- 构建权重 + 状态 + 前向 ----
    GGUFModelWeights weights;
    check(weights.build(model), "weights.build 成功");

    GGMLModelState st;
    st.init(weights, 8);

    std::vector<float> l1, l2;
    GGMLForward(weights, st, 0, 0, l1);
    check(l1.size() == 5, "logits 维度 = 词汇表大小(5)");
    bool fin = true;
    for (float v : l1)
        if (!std::isfinite(v))
            fin = false;
    check(fin, "logits 为有限值");

    GGMLModelState st2;
    st2.init(weights, 8);
    GGMLForward(weights, st2, 0, 0, l2);
    bool same = true;
    for (std::size_t i = 0; i < l1.size(); ++i)
        if (l1[i] != l2[i])
            same = false;
    check(same, "相同输入 → logits 确定性一致");

    // 不同 token → 不同 logits
    GGMLModelState st3;
    st3.init(weights, 8);
    std::vector<float> l3;
    GGMLForward(weights, st3, 1, 0, l3);
    bool diff = false;
    for (std::size_t i = 0; i < l1.size(); ++i)
        if (l1[i] != l3[i])
            diff = true;
    check(diff, "不同 token → logits 不同");

    // 自回归两 token：状态累积，第二 token 的 logits 与"从 pos=0 单跑"不同
    GGMLModelState st4;
    st4.init(weights, 8);
    std::vector<float> l_t0, l_t1, l_t1_only;
    GGMLForward(weights, st4, 0, 0, l_t0);
    GGMLForward(weights, st4, 1, 1, l_t1); // 同一状态，第二个 token（pos=1）
    GGMLModelState st5;
    st5.init(weights, 8);
    GGMLForward(weights, st5, 1, 0, l_t1_only); // 单独从 pos=0 跑 token 1
    bool accum = false;
    for (std::size_t i = 0; i < l_t1.size(); ++i)
        if (l_t1[i] != l_t1_only[i])
            accum = true;
    check(accum, "状态累积：pos=1 的 logits ≠ 单独 pos=0（KV/SSM 历史生效）");

    std::cout << (g_fail == 0 ? "\n✅ 全部通过" : "\n❌ 有失败") << std::endl;
    return g_fail == 0 ? 0 : 1;
}
