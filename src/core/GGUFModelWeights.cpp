/*
 * GGUFModelWeights.cpp — 模型权重加载（实现）
 *
 * 组装流程：
 *  1) 遍历张量信息表，把每个张量包装成 GGUFTensorView（data = 数据区指针 + offset）
 *  2) 从元数据解析 qwen35.* 配置
 *  3) 逐层组装 GGUFBlockWeights（按名称取各权重，缺失为 nullptr）
 */

#include "GGUFModelWeights.hpp"

#include <cstddef>
#include <utility>

#include "GGMLDequantize.hpp"

// ---------------------------------------------------------------------------
// 元数据小工具：按 key 查找并读取基础类型
// ---------------------------------------------------------------------------

namespace {

// 查找元数据值
const MetadataValue *find_meta(const GGUFModel &model, const std::string &key) {
    for (const auto &kv : model.metadata)
        if (kv.key == key)
            return &kv.value;
    return nullptr;
}

// 读 uint32（兼容 INT32 / UINT32 / BOOL）
bool get_u32(const GGUFModel &model, const std::string &key, std::uint32_t &out) {
    const auto *v = find_meta(model, key);
    if (!v)
        return false;
    if (const auto *u = std::get_if<std::uint32_t>(v)) {
        out = *u;
        return true;
    }
    if (const auto *i = std::get_if<std::int32_t>(v)) {
        out = static_cast<std::uint32_t>(*i);
        return true;
    }
    return false;
}

// 读 float（兼容 FLOAT32 / FLOAT64）
bool get_f32(const GGUFModel &model, const std::string &key, float &out) {
    const auto *v = find_meta(model, key);
    if (!v)
        return false;
    if (const auto *f = std::get_if<float>(v)) {
        out = *f;
        return true;
    }
    if (const auto *d = std::get_if<double>(v)) {
        out = static_cast<float>(*d);
        return true;
    }
    return false;
}

// 读字符串
bool get_str(const GGUFModel &model, const std::string &key, std::string &out) {
    const auto *v = find_meta(model, key);
    if (!v)
        return false;
    if (const auto *s = std::get_if<std::string>(v)) {
        out = *s;
        return true;
    }
    return false;
}

} // namespace

// ---------------------------------------------------------------------------
// GGUFTensorView 实现
// ---------------------------------------------------------------------------

std::uint64_t GGUFTensorView::element_count() const {
    std::uint64_t n = 1;
    for (auto d : dims)
        n *= d;
    return n;
}

bool GGUFTensorView::read_element(std::uint64_t idx, float &out) const {
    if (!data)
        return false;
    return GGMLDequantizeOne(type, data, idx, out);
}

bool GGUFTensorView::read_all(std::vector<float> &out) const {
    if (!data)
        return false;
    out.resize(static_cast<std::size_t>(element_count()));
    return GGMLDequantize(type, data, element_count(), out.data());
}

// ---------------------------------------------------------------------------
// GGUFModelWeights 实现
// ---------------------------------------------------------------------------

bool GGUFModelWeights::build(const GGUFModel &model) {
    // 前置条件：数据区必须已通过 map_data 映射
    if (model.data.data_ptr == nullptr || model.tensors.empty())
        return false;

    // ① 建立名称 → 张量视图 索引（零拷贝：data 直接指向映射区）
    views_.clear();
    for (const auto &t : model.tensors) {
        GGUFTensorView view;
        view.name = t.name;
        view.dims = t.dimensions;
        view.type = t.data_type;
        view.data = model.data.data_ptr + t.offset; // offset 为数据区相对偏移
        views_[t.name] = std::move(view);
    }
    token_embd_ = find("token_embd.weight");
    output_norm_ = find("output_norm.weight");

    // ② 解析配置
    GGUFModelConfig &c = config_;
    get_str(model, "general.architecture", c.arch);
    get_u32(model, c.arch + ".block_count", c.block_count);
    get_u32(model, c.arch + ".context_length", c.context_length);
    get_u32(model, c.arch + ".embedding_length", c.embedding_length);
    get_u32(model, c.arch + ".feed_forward_length", c.feed_forward_length);
    get_u32(model, c.arch + ".attention.head_count", c.head_count);
    get_u32(model, c.arch + ".attention.head_count_kv", c.head_count_kv);
    get_u32(model, c.arch + ".attention.key_length", c.key_length);
    get_u32(model, c.arch + ".attention.value_length", c.value_length);
    get_f32(model, c.arch + ".attention.layer_norm_rms_epsilon", c.rms_eps);
    get_u32(model, c.arch + ".ssm.conv_kernel", c.ssm_conv_kernel);
    get_u32(model, c.arch + ".ssm.state_size", c.ssm_state_size);
    get_u32(model, c.arch + ".ssm.group_count", c.ssm_group_count);
    get_u32(model, c.arch + ".ssm.time_step_rank", c.ssm_time_step_rank);
    get_u32(model, c.arch + ".ssm.inner_size", c.ssm_inner_size);
    get_u32(model, c.arch + ".full_attention_interval", c.full_attention_interval);
    get_u32(model, c.arch + ".rope.dimension_count", c.rope_dimension_count);
    get_f32(model, c.arch + ".rope.freq_base", c.rope_freq_base);

    // ③ 逐层组装权重
    blocks_.clear();
    const auto at = [&](std::uint32_t layer, const std::string &suffix) -> const GGUFTensorView * {
        return find("blk." + std::to_string(layer) + "." + suffix);
    };
    for (std::uint32_t layer = 0; layer < c.block_count; ++layer) {
        GGUFBlockWeights b;
        // 公共
        b.attn_norm = at(layer, "attn_norm.weight");
        b.post_attention_norm = at(layer, "post_attention_norm.weight");
        b.ffn_down = at(layer, "ffn_down.weight");
        b.ffn_gate = at(layer, "ffn_gate.weight");
        b.ffn_up = at(layer, "ffn_up.weight");
        // Attention 层
        b.attn_q = at(layer, "attn_q.weight");
        b.attn_k = at(layer, "attn_k.weight");
        b.attn_v = at(layer, "attn_v.weight");
        b.attn_output = at(layer, "attn_output.weight");
        b.attn_q_norm = at(layer, "attn_q_norm.weight");
        b.attn_k_norm = at(layer, "attn_k_norm.weight");
        // SSM 混合层
        b.ssm_a = at(layer, "ssm_a");
        b.ssm_conv1d = at(layer, "ssm_conv1d.weight");
        b.ssm_dt_bias = at(layer, "ssm_dt.bias");
        b.ssm_alpha = at(layer, "ssm_alpha.weight");
        b.ssm_beta = at(layer, "ssm_beta.weight");
        b.ssm_norm = at(layer, "ssm_norm.weight");
        b.ssm_out = at(layer, "ssm_out.weight");
        b.attn_qkv = at(layer, "attn_qkv.weight");
        b.attn_gate = at(layer, "attn_gate.weight");
        blocks_.push_back(b);
    }
    return true;
}

const GGUFTensorView *GGUFModelWeights::find(const std::string &name) const {
    const auto it = views_.find(name);
    return it != views_.end() ? &it->second : nullptr;
}
