/*
 * test_model_weights.cpp — 模型权重加载单元测试（不依赖真实模型）
 *
 * 手建一个最小 GGUFModel（假配置 + 合成张量 + 假数据区），验证：
 *  - 按名索引 find / 未找到返回 nullptr
 *  - 配置解析（qwen35.* 字段）
 *  - 逐层权重组装（SSM 层 vs Attention 层）
 *  - 反量化读取 read_element / read_all
 */

#include <initializer_list>
#include <iostream>
#include <vector>

#include "GGUFModelWeights.hpp"

static int g_fail = 0;

static void check(bool ok, const char *msg) {
    if (ok)
        std::cout << "  ✅ " << msg << std::endl;
    else {
        std::cout << "  ❌ " << msg << std::endl;
        ++g_fail;
    }
}

// 给假模型加一条元数据
static void add_str(GGUFModel &m, const std::string &key, const std::string &v) {
    m.metadata.push_back({key, GGUFValueType::STRING, v});
}
static void add_u32(GGUFModel &m, const std::string &key, std::uint32_t v) {
    m.metadata.push_back({key, GGUFValueType::UINT32, v});
}
static void add_f32(GGUFModel &m, const std::string &key, float v) {
    m.metadata.push_back({key, GGUFValueType::FLOAT32, v});
}

// 给假模型加一个 F32 张量（offset 指向假数据区）
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
    std::cout << "=== 模型权重加载单元测试 ===" << std::endl;

    // ---- 构造假模型：blk.0 = SSM 层，blk.1 = Attention 层 ----
    GGUFModel model;
    std::vector<float> fake(1024, 0.0f); // 假数据区
    for (int i = 0; i < 8; ++i)
        fake[static_cast<std::size_t>(i)] = static_cast<float>(i + 1); // 1..8
    model.data.data_ptr = reinterpret_cast<const std::uint8_t *>(fake.data());
    model.data.data_offset = 0;
    model.data.data_size = fake.size() * sizeof(float);

    // 配置元数据（qwen35 架构）
    add_str(model, "general.architecture", "qwen35");
    add_u32(model, "qwen35.block_count", 2);
    add_u32(model, "qwen35.context_length", 262144);
    add_u32(model, "qwen35.embedding_length", 1024);
    add_u32(model, "qwen35.feed_forward_length", 3584);
    add_u32(model, "qwen35.attention.head_count", 8);
    add_u32(model, "qwen35.attention.head_count_kv", 2);
    add_u32(model, "qwen35.attention.key_length", 256);
    add_u32(model, "qwen35.attention.value_length", 256);
    add_f32(model, "qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
    add_u32(model, "qwen35.ssm.conv_kernel", 4);
    add_u32(model, "qwen35.ssm.state_size", 128);
    add_u32(model, "qwen35.ssm.group_count", 16);
    add_u32(model, "qwen35.ssm.time_step_rank", 16);
    add_u32(model, "qwen35.ssm.inner_size", 2048);
    add_u32(model, "qwen35.full_attention_interval", 4);
    add_u32(model, "qwen35.rope.dimension_count", 64);
    add_f32(model, "qwen35.rope.freq_base", 10000000.0f);

    // 张量表（offset 依次累加，模拟数据区布局）
    std::uint64_t off = 0;
    add_tensor(model, "token_embd.weight", {4, 2}, off);
    off += 8;
    add_tensor(model, "output_norm.weight", {4}, off);
    off += 4;
    // blk.0：SSM 混合层
    add_tensor(model, "blk.0.attn_norm.weight", {4}, off);
    off += 4;
    add_tensor(model, "blk.0.ssm_a", {2}, off);
    off += 2;
    add_tensor(model, "blk.0.ssm_conv1d.weight", {4, 3}, off);
    off += 12;
    add_tensor(model, "blk.0.ssm_dt.bias", {2}, off);
    off += 2;
    add_tensor(model, "blk.0.ssm_alpha.weight", {4, 2}, off);
    off += 8;
    add_tensor(model, "blk.0.ssm_beta.weight", {4, 2}, off);
    off += 8;
    add_tensor(model, "blk.0.attn_qkv.weight", {4, 6}, off);
    off += 24;
    add_tensor(model, "blk.0.attn_gate.weight", {4, 2}, off);
    off += 8;
    add_tensor(model, "blk.0.ssm_norm.weight", {8}, off);
    off += 8;
    add_tensor(model, "blk.0.ssm_out.weight", {2, 4}, off);
    off += 8;
    add_tensor(model, "blk.0.ffn_down.weight", {2, 4}, off);
    off += 8;
    add_tensor(model, "blk.0.ffn_gate.weight", {4, 2}, off);
    off += 8;
    add_tensor(model, "blk.0.ffn_up.weight", {4, 2}, off);
    off += 8;
    add_tensor(model, "blk.0.post_attention_norm.weight", {4}, off);
    off += 4;
    // blk.1：纯 Attention 层
    add_tensor(model, "blk.1.attn_norm.weight", {4}, off);
    off += 4;
    add_tensor(model, "blk.1.attn_q.weight", {4, 8}, off);
    off += 32;
    add_tensor(model, "blk.1.attn_k.weight", {4, 2}, off);
    off += 8;
    add_tensor(model, "blk.1.attn_v.weight", {4, 2}, off);
    off += 8;
    add_tensor(model, "blk.1.attn_output.weight", {8, 4}, off);
    off += 32;
    add_tensor(model, "blk.1.attn_q_norm.weight", {4}, off);
    off += 4;
    add_tensor(model, "blk.1.attn_k_norm.weight", {4}, off);
    off += 4;
    add_tensor(model, "blk.1.ffn_down.weight", {2, 4}, off);
    off += 8;
    add_tensor(model, "blk.1.ffn_gate.weight", {4, 2}, off);
    off += 8;
    add_tensor(model, "blk.1.ffn_up.weight", {4, 2}, off);
    off += 8;
    add_tensor(model, "blk.1.post_attention_norm.weight", {4}, off);
    off += 4;

    // ---- 构建并断言 ----
    GGUFModelWeights w;
    check(w.build(model), "build 成功");
    check(w.count() == model.tensors.size(), "索引张量数量一致");

    const auto *embd = w.find("token_embd.weight");
    check(embd != nullptr, "find('token_embd.weight') 命中");
    check(embd && embd->dims.size() == 2 && embd->dims[0] == 4 && embd->dims[1] == 2,
          "token_embd 维度 {4,2} 正确");
    check(w.find("no.such.tensor") == nullptr, "find(不存在) 返回 nullptr");
    check(w.token_embd() == embd, "token_embd() 快捷访问一致");
    check(w.output_norm() != nullptr, "output_norm() 存在");

    const auto &cfg = w.config();
    check(cfg.arch == "qwen35", "架构=qwen35");
    check(cfg.block_count == 2, "block_count=2");
    check(cfg.embedding_length == 1024, "embedding_length=1024");
    check(cfg.feed_forward_length == 3584, "feed_forward_length=3584");
    check(cfg.head_count == 8 && cfg.head_count_kv == 2, "head=8, kv=2");
    check(cfg.ssm_state_size == 128, "ssm_state_size=128");
    check(cfg.full_attention_interval == 4, "full_attention_interval=4");

    check(w.blocks().size() == 2, "块数量=2");
    const auto &b0 = w.blocks()[0];
    check(b0.is_ssm() && !b0.is_attention(), "blk.0 是 SSM 层");
    check(b0.attn_norm && b0.post_attention_norm && b0.ffn_down && b0.ffn_gate && b0.ffn_up,
          "blk.0 公共权重齐全");
    check(b0.ssm_a && b0.ssm_conv1d && b0.ssm_dt_bias && b0.attn_qkv && b0.attn_gate &&
              b0.ssm_norm && b0.ssm_out,
          "blk.0 SSM 权重齐全");
    check(b0.attn_q == nullptr, "blk.0 无 attn_q（非 attention 层）");

    const auto &b1 = w.blocks()[1];
    check(b1.is_attention() && !b1.is_ssm(), "blk.1 是 Attention 层");
    check(b1.attn_q && b1.attn_k && b1.attn_v && b1.attn_output && b1.attn_q_norm && b1.attn_k_norm,
          "blk.1 attention 权重齐全");
    check(b1.ssm_a == nullptr && b1.attn_qkv == nullptr, "blk.1 无 SSM 权重");

    // 反量化读取（F32）
    float v = 0;
    check(embd->read_element(0, v) && v == 1.0f, "read_element(0)=1.0");
    check(embd->read_element(7, v) && v == 8.0f, "read_element(7)=8.0");
    std::vector<float> buf;
    check(embd->read_all(buf) && buf.size() == 8 && buf[3] == 4.0f,
          "read_all 返回 8 个元素且 buf[3]=4.0");

    std::cout << (g_fail == 0 ? "\n✅ 全部通过" : "\n❌ 有失败") << std::endl;
    return g_fail == 0 ? 0 : 1;
}
