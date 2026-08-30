/*
 * test_generate.cpp — 阶段⑥ 第 2 步：自回归生成循环单元测试
 *
 * 复用 2 层假模型（blk.0 SSM + blk.1 Attention），验证：
 *  - 给定 prompt 生成指定数量的新 token
 *  - 相同 seed + greedy → 生成结果确定性一致
 *  - eos 触发提前停止
 */

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

#include "GGMLGenerate.hpp"

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

// 构造 2 层假模型（与 test_forward 相同的结构）
static GGUFModel make_fake_model() {
    GGUFModel model;
    std::vector<float> fake(4096);
    for (std::size_t i = 0; i < fake.size(); ++i)
        fake[i] = 0.1f * static_cast<float>((i % 7) + 1);
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
    return model;
}

int main() {
    std::cout << "=== 阶段⑥ 第2步：自回归生成循环单元测试 ===" << std::endl;

    GGUFModelWeights weights;
    if (!weights.build(make_fake_model())) {
        check(false, "weights.build 失败");
        return 1;
    }

    GGMLSampler sampler;
    sampler.mode = GGMLSampleMode::GREEDY; // 确定性采样

    // [1] 生成 3 个新 token
    GGMLModelState st;
    st.init(weights, 8);
    const std::vector<int> prompt = {0, 1};
    const auto out = GGMLGenerate(weights, st, sampler, prompt, 3, -1);
    check(out.size() == 3, "prompt=[0,1] 生成 3 个新 token");
    bool valid = true;
    for (int t : out)
        if (t < 0 || t >= 5)
            valid = false;
    check(valid, "生成的 token id 在词汇表范围内");

    // [2] 确定性：相同 seed + greedy → 相同结果
    GGMLSampler sampler2;
    sampler2.mode = GGMLSampleMode::GREEDY;
    GGMLModelState st2;
    st2.init(weights, 8);
    const auto out2 = GGMLGenerate(weights, st2, sampler2, prompt, 3, -1);
    check(out == out2, "greedy 生成结果确定一致");

    // [3] eos 触发提前停止：eos=首个生成 token 时，greedy 下第一个 token 即命中 → 空输出
    GGMLSampler sampler3;
    sampler3.mode = GGMLSampleMode::GREEDY;
    GGMLModelState st3;
    st3.init(weights, 8);
    const auto out3 = GGMLGenerate(weights, st3, sampler3, prompt, 3, out[0]);
    check(out3.empty(), "eos=首个生成 token → 命中即停（eos 不输出，结果为空）");

    std::cout << (g_fail == 0 ? "\n✅ 全部通过" : "\n❌ 有失败") << std::endl;
    return g_fail == 0 ? 0 : 1;
}
