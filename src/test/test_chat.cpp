/*
 * test_chat.cpp — 阶段⑥ 第 3 步：多轮对话单元测试
 *
 * 用 2 层假模型 + 手建假分词器（vocab = a-e），验证：
 *  - chat 返回非空回复
 *  - 回复 token 在词汇表范围内
 *  - 多轮对话历史累积（每轮都能正常生成）
 *  - clear 后重新开始
 */

#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

#include "GGMLChat.hpp"

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

// 2 层假模型（与 test_forward 相同结构，vocab=5）
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
    std::cout << "=== 阶段⑥ 第3步：多轮对话单元测试 ===" << std::endl;

    GGUFModelWeights weights;
    if (!weights.build(make_fake_model())) {
        check(false, "weights.build 失败");
        return 1;
    }

    // 假分词器：vocab = {a,b,c,d,e}（token id 0-4）
    GGUFTokenizer tok;
    tok.tokens = {"a", "b", "c", "d", "e"};
    tok.token_types = {1, 1, 1, 1, 1};
    tok.rebuild_index();

    GGMLChat chat;
    chat.init(weights, tok, GGMLSampleMode::GREEDY);

    const std::string r1 = chat.chat("a", 4);
    check(!r1.empty(), "第 1 轮 chat 返回非空回复");
    bool valid = true;
    for (char c : r1)
        if (c < 'a' || c > 'e')
            valid = false;
    check(valid, "回复字符都在词汇表 a-e 内");

    const std::string r2 = chat.chat("b", 4);
    check(!r2.empty(), "第 2 轮 chat（历史累积）返回非空回复");

    chat.clear();
    const std::string r3 = chat.chat("c", 4);
    check(!r3.empty(), "clear 后重新开始对话正常");

    std::cout << (g_fail == 0 ? "\n✅ 全部通过" : "\n❌ 有失败") << std::endl;
    return g_fail == 0 ? 0 : 1;
}
