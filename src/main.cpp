
/*
 * main.cpp — GGUF 解析与模型计算全流程演示
 *
 * 用法: ./main [model.gguf]
 *   缺省模型: /home/dongfan/llm/Qwen3.5-0.8B-clean-BF16.gguf
 *
 * 演示流程（对应 README 各阶段）：
 *   ① 文件头     ② 元数据 KV    ③ 张量信息表 + 数据区
 *   类型系统验证  mmap 直接读取   反量化（含 F32 交叉验证）
 *   ④ 权重索引   ⑤ 基础算子 / Attention 层 / SSM 层 / 全模型前向
 *   ⑥ 生成引擎   多轮对话封装     分词器（字节级 BPE 往返）
 *
 * 结构说明：每个演示段拆成独立函数（demo_*），main() 只做编排；
 *   mmap 由 DataMapGuard（RAII）管理，不再手动 map/unmap 重复调用。
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "GGMLChat.hpp"
#include "GGMLDequantize.hpp"
#include "GGMLForward.hpp"
#include "GGMLGenerate.hpp"
#include "GGMLNorm.hpp"
#include "GGMLRope.hpp"
#include "GGMLSSM.hpp"
#include "GGMLSampler.hpp"
#include "GGMLTransformer.hpp"
#include "GGMLType.hpp"
#include "GGUFLoader.hpp"
#include "GGUFModelWeights.hpp"
#include "GGUFTokenizer.hpp"

namespace {

constexpr const char *kDefaultModelPath = "/home/dongfan/llm/Qwen3.5-0.8B-clean-BF16.gguf";

// ===========================================================================
// 小工具
// ===========================================================================

// RAII：构造时 map_data、析构时 unmap_data，保证演示期间映射始终有效
class DataMapGuard {
  public:
    DataMapGuard(GGUFModel &model, const std::string &path) : model_(model), path_(path) {
        if (!GGUFLoader::map_data(path_, model_))
            std::cerr << "  ❌ map_data 失败" << std::endl;
    }
    ~DataMapGuard() {
        GGUFLoader::unmap_data(model_);
    }
    DataMapGuard(const DataMapGuard &) = delete;
    DataMapGuard &operator=(const DataMapGuard &) = delete;

    bool ok() const {
        return model_.data.data_ptr != nullptr;
    }

  private:
    GGUFModel &model_;
    std::string path_;
};

// 按类型名查找张量（对类型编号校准不敏感：BF16 在本模型为 30，标准为 21）
const GGUFTensorInfo *find_tensor_of_type(const GGUFModel &model, const char *type_name) {
    for (const auto &t : model.tensors)
        if (std::string(GGMLTypeName(t.data_type)) == type_name)
            return &t;
    return nullptr;
}

// 统计 NaN 个数
std::size_t count_nan(const std::vector<float> &v) {
    std::size_t n = 0;
    for (float f : v)
        if (std::isnan(f))
            ++n;
    return n;
}

// 统计非有限（NaN/Inf）个数
std::size_t count_nonfinite(const std::vector<float> &v) {
    std::size_t n = 0;
    for (float f : v)
        if (!std::isfinite(f))
            ++n;
    return n;
}

// 打印 float 向量前 n 个元素（label 形如 "token_embd.weight"）
void print_floats(const std::string &label, const std::vector<float> &v, std::size_t n) {
    const std::size_t m = std::min(n, v.size());
    std::cout << "  " << label << " 前 " << m << " 个元素: ";
    for (std::size_t i = 0; i < m; ++i)
        std::cout << v[i] << (i + 1 < m ? ", " : "");
    std::cout << std::endl;
}

// 找第一种指定块（want_attention=true 找 Attention 层，否则找 SSM 层），返回层下标
const GGUFBlockWeights *find_block(const GGUFModelWeights &w, bool want_attention, int &layer) {
    for (std::size_t i = 0; i < w.blocks().size(); ++i) {
        const auto &b = w.blocks()[i];
        if (want_attention ? b.is_attention() : b.is_ssm()) {
            layer = static_cast<int>(i);
            return &b;
        }
    }
    return nullptr;
}

// ===========================================================================
// ① 文件头 / ② 元数据 / ③ 张量表 + 数据区
// ===========================================================================

void demo_header(const GGUFModel &model) {
    std::cout << "✅ 加载成功" << std::endl;
    std::cout << "  魔数: 0x" << std::hex << model.header.magic << std::dec << std::endl;
    std::cout << "  版本: " << model.header.version << std::endl;
    std::cout << "  元数据 KV 数量: " << model.header.metadata_kv_count << std::endl;
    std::cout << "  张量数量: " << model.header.tensor_count << std::endl;
}

void demo_metadata(const GGUFModel &model) {
    std::cout << "\n=== 元数据 KV 列表（前 20 条）===" << std::endl;
    const std::size_t n = std::min<std::size_t>(model.metadata.size(), 20);
    for (std::size_t i = 0; i < n; ++i) {
        const auto &kv = model.metadata[i];
        std::cout << "  [" << GGUFValueTypeName(kv.value_type) << "] " << kv.key << " = ";
        printMetadataValue(std::cout, kv.value);
        std::cout << std::endl;
    }
    if (model.metadata.size() > n)
        std::cout << "  ... 共 " << model.metadata.size() << " 条" << std::endl;
}

void demo_tensor_table(const GGUFModel &model) {
    std::cout << "\n=== 张量信息表（前 10 个）===" << std::endl;
    const std::size_t n = std::min<std::size_t>(model.tensors.size(), 10);
    for (std::size_t i = 0; i < n; ++i) {
        const auto &t = model.tensors[i];
        std::cout << "  " << t.name << "  dims=[";
        for (std::size_t d = 0; d < t.dimensions.size(); ++d) {
            if (d)
                std::cout << ", ";
            std::cout << t.dimensions[d];
        }
        std::cout << "]  type=" << t.data_type << "(" << GGMLTypeName(t.data_type)
                  << ")  offset=" << t.offset << "  elements=" << t.element_count() << std::endl;
    }
    if (model.tensors.size() > n)
        std::cout << "  ... 共 " << model.tensors.size() << " 个张量" << std::endl;
}

void demo_data_region(const GGUFModel &model) {
    std::cout << "\n=== 张量数据区 ===" << std::endl;
    std::cout << "  起始偏移: " << model.data.data_offset << std::endl;
    std::cout << "  总大小: " << model.data.data_size << " 字节 (" << std::fixed
              << std::setprecision(2)
              << static_cast<double>(model.data.data_size) / (1024.0 * 1024.0) << " MiB)"
              << std::endl;
    std::cout << "  数据指针: " << (model.data.data_ptr ? "已挂载" : "未挂载（延迟加载，待 mmap）")
              << std::endl;
}

// ===========================================================================
// 类型系统验证 + 反推 字节/元素
// ===========================================================================

void demo_type_verification(const GGUFModel &model) {
    if (model.tensors.empty())
        return;
    std::cout << "\n=== GGML 类型系统验证 ===" << std::endl;

    // ① 按类型计算每个张量的字节数并累加
    std::uint64_t total_bytes = 0;
    for (const auto &t : model.tensors)
        total_bytes += GGMLBytes(t.data_type, t.element_count());

    // ② 最后一个张量之后可能还有文件尾部填充（不属于任何张量）
    const auto &last = model.tensors.back();
    const std::uint64_t last_end = last.offset + GGMLBytes(last.data_type, last.element_count());
    const std::uint64_t trailing =
        (last_end <= model.data.data_size) ? model.data.data_size - last_end : 0;

    std::cout << "  ① 张量数据累加: " << total_bytes << std::endl;
    std::cout << "  ② 尾部填充: " << trailing << std::endl;
    std::cout << "  数据区实际大小: " << model.data.data_size << std::endl;
    std::cout << "  ① + ② 与文件吻合: "
              << ((total_bytes + trailing) == model.data.data_size ? "✅ 完全一致" : "❌ 不一致")
              << std::endl;

    // 诊断：从张量 offset 反推每种 data_type 的真实 字节/元素（校准类型表用）
    // 注意：tensor.offset 是相对数据区起点的偏移，范围 [0, data_size)
    std::cout << "\n=== 类型诊断：反推 字节/元素 ===" << std::endl;
    std::map<std::uint32_t, std::pair<std::uint64_t, std::uint64_t>> stat;
    for (std::size_t i = 0; i < model.tensors.size(); ++i) {
        const auto &t = model.tensors[i];
        const std::uint64_t end = (i + 1 < model.tensors.size())
                                      ? model.tensors[i + 1].offset
                                      : model.data.data_size; // 最后一个到数据区末尾
        stat[t.data_type].first += end - t.offset;
        stat[t.data_type].second += t.element_count();
    }
    for (const auto &[type, p] : stat)
        std::cout << "  type=" << type << "(" << GGMLTypeName(type) << ")"
                  << "  字节/元素=" << static_cast<double>(p.first) / static_cast<double>(p.second)
                  << "  总字节=" << p.first << std::endl;
}

// ===========================================================================
// mmap 直接读取 / 反量化演示（每个演示内部 RAII 管理映射）
// ===========================================================================

void demo_direct_read(GGUFModel &model, const std::string &path) {
    std::cout << "\n=== mmap 映射后读取张量数据 ===" << std::endl;
    DataMapGuard guard(model, path);
    if (!guard.ok())
        return;
    std::cout << "  映射成功: data_ptr = " << static_cast<const void *>(model.data.data_ptr)
              << "（映射 " << model.data.map_len << " 字节）" << std::endl;

    // 找一个 F32 张量演示直接读取
    if (const auto *target = find_tensor_of_type(model, "F32")) {
        // GGUF 的 tensor.offset 是相对"数据区起点"的偏移，
        // 而 data_ptr 正指向数据区起点 → 张量位置 = data_ptr + offset
        const float *f = reinterpret_cast<const float *>(model.data.data_ptr + target->offset);
        std::cout << "  张量: " << target->name << "  elements=" << target->element_count()
                  << std::endl;
        std::cout << "  前 8 个值: ";
        const std::size_t n = std::min<std::size_t>(target->element_count(), 8);
        for (std::size_t i = 0; i < n; ++i)
            std::cout << f[i] << (i + 1 < n ? ", " : "");
        std::cout << std::endl;
    } else {
        std::cout << "  未找到 F32 张量" << std::endl;
    }
}

void demo_dequantize(GGUFModel &model, const std::string &path) {
    std::cout << "\n=== 反量化演示（GGMLDequantize）===" << std::endl;
    DataMapGuard guard(model, path);
    if (!guard.ok())
        return;

    // BF16：按类型名查找（兼容类型编号校准，本模型为 30）
    if (const auto *bf16 = find_tensor_of_type(model, "BF16")) {
        const std::uint8_t *raw = model.data.data_ptr + bf16->offset;
        std::cout << "  BF16 张量: " << bf16->name << "  elements=" << bf16->element_count()
                  << std::endl;
        // 转换函数自检：BF16 0x3F80=1.0、0x4000=2.0
        std::cout << "  转换自检: GGMLBF16ToFloat(0x3F80)=" << GGMLBF16ToFloat(0x3F80)
                  << "  (0x4000)=" << GGMLBF16ToFloat(0x4000) << std::endl;
        std::cout << "  反量化前 8 个值: ";
        const std::size_t n = std::min<std::size_t>(bf16->element_count(), 8);
        for (std::size_t i = 0; i < n; ++i) {
            float v = 0;
            GGMLDequantizeOne(bf16->data_type, raw, i, v);
            std::uint16_t bits = 0;
            std::memcpy(&bits, raw + i * 2, 2);
            std::cout << v << "(0x" << std::hex << bits << std::dec << ")"
                      << (i + 1 < n ? ", " : "");
        }
        std::cout << std::endl;
    } else {
        std::cout << "  未找到 BF16 张量" << std::endl;
    }

    // 交叉验证：F32 张量 "直接读" vs "反量化" 应完全一致
    if (const auto *f32 = find_tensor_of_type(model, "F32")) {
        const std::uint8_t *raw = model.data.data_ptr + f32->offset;
        float deq[8] = {};
        GGMLDequantize(f32->data_type, raw, 8, deq);
        const float *direct = reinterpret_cast<const float *>(raw);
        bool same = true;
        for (std::size_t i = 0; i < 8; ++i)
            if (deq[i] != direct[i])
                same = false;
        std::cout << "  F32 交叉验证(反量化==直接读): " << (same ? "✅ 一致" : "❌ 不一致")
                  << std::endl;
    }
}

// ===========================================================================
// ④ 权重索引 + ⑤ 基础算子演示
// ===========================================================================

void demo_weights(const GGUFModelWeights &weights) {
    const auto &cfg = weights.config();
    std::cout << "  架构: " << (cfg.arch.empty() ? "?" : cfg.arch) << "  层数: " << cfg.block_count
              << "  hidden: " << cfg.embedding_length << "  heads: " << cfg.head_count << "x"
              << cfg.head_count_kv << "kv  FFN: " << cfg.feed_forward_length
              << "  SSM state: " << cfg.ssm_state_size
              << "  全注意力间隔: " << cfg.full_attention_interval << std::endl;
    std::cout << "  张量视图数: " << weights.count() << std::endl;

    std::uint32_t attn = 0, ssm = 0;
    for (const auto &b : weights.blocks())
        (b.is_attention() ? attn : ssm)++;
    std::cout << "  块类型: Attention 层 " << attn << " 个 / SSM 层 " << ssm << " 个" << std::endl;

    if (const auto *embd = weights.token_embd()) {
        std::vector<float> buf;
        if (embd->read_all(buf))
            print_floats("token_embd.weight", buf, 8);
    }
    if (!weights.blocks().empty()) {
        const auto &b0 = weights.blocks()[0];
        const char *kind = b0.is_ssm() ? "SSM 混合层" : "Attention 层";
        if (const auto *norm = b0.attn_norm) {
            std::vector<float> buf;
            if (norm->read_all(buf))
                print_floats("blk.0(" + std::string(kind) + ") attn_norm", buf, 4);
        }
    }
}

void demo_basic_ops(const GGUFModelWeights &weights) {
    const auto &cfg = weights.config();
    const int hidden = static_cast<int>(cfg.embedding_length);
    if (weights.blocks().empty())
        return;
    const auto &b0 = weights.blocks()[0];

    // RMSNorm：输入全 1，用真实 gamma
    if (const auto *norm = b0.attn_norm) {
        std::vector<float> gamma;
        if (norm->read_all(gamma)) {
            std::vector<float> x(static_cast<std::size_t>(hidden), 1.0f);
            std::vector<float> y(static_cast<std::size_t>(hidden));
            GGMLRmsNorm(x.data(), gamma.data(), y.data(), hidden, cfg.rms_eps);
            print_floats("RMSNorm(全1输入)", y, 4);
        }
    }
    // SwiGLU FFN：诊断权重中的 NaN（提示随统计结果动态变化）
    if (b0.ffn_gate && b0.ffn_up && b0.ffn_down) {
        std::vector<float> gate, up, down;
        if (b0.ffn_gate->read_all(gate) && b0.ffn_up->read_all(up) && b0.ffn_down->read_all(down)) {
            const std::size_t total = count_nan(gate) + count_nan(up) + count_nan(down);
            std::cout << "  SwiGLU 权重: gate=" << gate.size() << " up=" << up.size()
                      << " down=" << down.size() << "  NaN 数: " << count_nan(gate) << "/"
                      << count_nan(up) << "/" << count_nan(down)
                      << (total > 0 ? "（BF16 权重本身含 NaN）" : "（权重干净，无 NaN）")
                      << std::endl;
        }
    }
    // RoPE：旋转保范演示
    {
        std::vector<float> q(8), y(8);
        for (int i = 0; i < 8; ++i)
            q[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
        GGMLRopeNeox(q.data(), y.data(), 8, 8, 3, cfg.rope_freq_base);
        float n0 = 0, n1 = 0;
        for (int i = 0; i < 4; ++i) {
            n0 += q[static_cast<std::size_t>(i)] * q[static_cast<std::size_t>(i)] +
                  q[static_cast<std::size_t>(i + 4)] * q[static_cast<std::size_t>(i + 4)];
            n1 += y[static_cast<std::size_t>(i)] * y[static_cast<std::size_t>(i)] +
                  y[static_cast<std::size_t>(i + 4)] * y[static_cast<std::size_t>(i + 4)];
        }
        std::cout << "  RoPE(pos=3) 前 4 维: ";
        for (int i = 0; i < 4; ++i)
            std::cout << y[static_cast<std::size_t>(i)] << (i + 1 < 4 ? ", " : "");
        std::cout << "  范数 " << n0 << "→" << n1 << "（应相等）" << std::endl;
    }
}

// ===========================================================================
// ⑤ 第2~4 步：Attention 层 / SSM 层 / 全模型前向
// ===========================================================================

void demo_attention_block(const GGUFModelWeights &weights) {
    int layer = -1;
    const GGUFBlockWeights *blk = find_block(weights, /*attention=*/true, layer);
    const auto *embd = weights.token_embd();
    if (!blk || !embd) {
        std::cout << "  未找到 Attention 层或 token_embd" << std::endl;
        return;
    }
    const auto &cfg = weights.config();
    const int hidden = static_cast<int>(cfg.embedding_length);
    // 用 token 0 的 embedding 作为输入（读前 hidden 个元素）
    std::vector<float> x(static_cast<std::size_t>(hidden));
    for (int i = 0; i < hidden; ++i)
        embd->read_element(static_cast<std::uint64_t>(i), x[static_cast<std::size_t>(i)]);
    // KV cache：2 kv 头，head_dim 256，最多 8 位置
    GGMLKVCache cache;
    cache.init(static_cast<int>(cfg.head_count_kv), static_cast<int>(cfg.key_length),
               static_cast<int>(cfg.value_length), 8);
    std::vector<float> y(static_cast<std::size_t>(hidden));
    GGMLTransformerAttentionBlock(*blk, cfg, cache, 0, x.data(), y.data());
    print_floats("blk." + std::to_string(layer) + "(Attention 层) 前向输出", y, 4);
    std::cout << "  全有限: " << (count_nonfinite(y) == 0 ? "✅" : "❌（模型权重含 NaN）")
              << std::endl;
}

void demo_ssm_block(const GGUFModelWeights &weights) {
    int layer = -1;
    const GGUFBlockWeights *blk = find_block(weights, /*attention=*/false, layer);
    const auto *embd = weights.token_embd();
    if (!blk || !embd) {
        std::cout << "  未找到 SSM 层或 token_embd" << std::endl;
        return;
    }
    const auto &cfg = weights.config();
    const int hidden = static_cast<int>(cfg.embedding_length);
    std::vector<float> x(static_cast<std::size_t>(hidden));
    for (int i = 0; i < hidden; ++i)
        embd->read_element(static_cast<std::uint64_t>(i), x[static_cast<std::size_t>(i)]);
    // SSM 状态：n_group 个 v-head × d_state×d_state；conv_dim = 2×key_dim + value_dim
    const int n_group = static_cast<int>(cfg.ssm_group_count);
    const int d_state = static_cast<int>(cfg.ssm_state_size);
    const int key_dim = d_state * n_group;
    const int conv_dim = 2 * key_dim + static_cast<int>(cfg.ssm_inner_size);
    GGMLSSMState st;
    st.init(n_group, d_state, conv_dim);
    std::vector<float> y(static_cast<std::size_t>(hidden));
    GGMLSSMLayer(*blk, cfg, st, x.data(), y.data());
    print_floats("blk." + std::to_string(layer) + "(SSM 层) 前向输出", y, 4);
    std::cout << "  全有限: " << (count_nonfinite(y) == 0 ? "✅" : "❌（模型权重含 NaN）")
              << std::endl;
}

void demo_full_forward(const GGUFModelWeights &weights) {
    GGMLModelState st;
    st.init(weights, 32);
    std::vector<float> logits;
    GGMLForward(weights, st, 0, 0, logits);
    const std::size_t bad = count_nonfinite(logits);
    print_floats("token 0 前向 → logits", logits, 5);
    std::cout << "  全有限: "
              << (bad == 0 ? "✅" : "❌（NaN/Inf " + std::to_string(bad) + " 个，模型权重含 NaN）")
              << std::endl;
}

// ===========================================================================
// ⑥ 生成引擎 / 多轮对话
// ===========================================================================

void demo_generate(const GGUFModel &model, const GGUFModelWeights &weights) {
    GGUFTokenizer tok;
    if (!tok.build_from(model)) {
        std::cout << "  tokenizer 构建失败" << std::endl;
        return;
    }
    const std::string prompt_text = "Hello";
    const auto prompt_tokens = tok.encode(prompt_text);
    if (prompt_tokens.empty()) {
        std::cout << "  prompt 编码为空" << std::endl;
        return;
    }
    GGMLModelState st;
    st.init(weights, 32);
    // 先探测一次 logits：模型含 NaN 则无法生成（避免采样崩溃）
    std::vector<float> probe;
    GGMLForward(weights, st, prompt_tokens[0], 0, probe);
    if (count_nonfinite(probe) > 0) {
        std::cout << "  ❌ 模型 logits 含 NaN（模型权重本身含 NaN），无法生成有效文本；"
                     "建议换干净的 Qwen3.5 GGUF"
                  << std::endl;
        return;
    }
    GGMLSampler sampler;
    sampler.mode = GGMLSampleMode::TOP_K_P;
    const auto gen = GGMLGenerate(weights, st, sampler, prompt_tokens, 16, tok.eos_id);
    std::string text = prompt_text;
    for (int t : gen)
        text += tok.decode(t);
    std::cout << "  prompt: '" << prompt_text << "'" << std::endl;
    std::cout << "  生成  : '" << text << "'  （" << gen.size() << " 个新 token）" << std::endl;
}

void demo_chat(const GGUFModel &model, const GGUFModelWeights &weights) {
    GGUFTokenizer tok;
    if (!tok.build_from(model)) {
        std::cout << "  tokenizer 构建失败" << std::endl;
        return;
    }
    GGMLChat chat;
    chat.init(weights, tok, GGMLSampleMode::TOP_K_P, 42, 32);
    std::cout << "  GGMLChat 已初始化（Qwen 风格多轮对话封装）" << std::endl;

    // 探测一次 logits：确认能否实际对话（提示随真实状态动态变化，避免误导）
    GGMLModelState st;
    st.init(weights, 32);
    std::vector<float> probe;
    GGMLForward(weights, st, 0, 0, probe);
    if (count_nonfinite(probe) > 0) {
        std::cout << "  ⚠️ 模型 logits 含 NaN（FFN 权重含 NaN），无法实际对话；"
                     "建议换干净的 Qwen3.5 GGUF 后体验完整生成链路"
                  << std::endl;
    } else {
        std::cout << "  ✅ 模型 logits 有限，可正常对话（本演示仅初始化，不自动对话）" << std::endl;
    }
}

// ===========================================================================
// ④⑤⑥ 组合入口（权重索引 + 算子 + 层前向 + 全模型 + 生成 + 对话）
// ===========================================================================

void demo_weights_and_forward(GGUFModel &model, const std::string &path) {
    std::cout << "\n=== 模型权重加载演示（阶段④）===" << std::endl;
    DataMapGuard guard(model, path);
    if (!guard.ok())
        return;

    GGUFModelWeights weights;
    if (!weights.build(model)) {
        std::cerr << "  ❌ 权重索引构建失败" << std::endl;
        return;
    }
    demo_weights(weights);

    std::cout << "\n  --- 阶段⑤ 基础算子演示（真实权重）---" << std::endl;
    demo_basic_ops(weights);

    std::cout << "\n  --- 阶段⑤ 第2步：Attention 层前向（真实权重）---" << std::endl;
    demo_attention_block(weights);

    std::cout << "\n  --- 阶段⑤ 第3步：SSM 混合层前向（真实权重）---" << std::endl;
    demo_ssm_block(weights);

    std::cout << "\n  --- 阶段⑤ 第4步：全模型前向（真实权重）---" << std::endl;
    demo_full_forward(weights);

    std::cout << "\n  --- 阶段⑥：生成引擎（真实模型）---" << std::endl;
    demo_generate(model, weights);

    std::cout << "\n  --- 阶段⑥：Chat 多轮对话（真实模型）---" << std::endl;
    demo_chat(model, weights);
}

// ===========================================================================
// Tokenizer 演示
// ===========================================================================

void demo_tokenizer(const GGUFModel &model) {
    std::cout << "\n=== Tokenizer 演示 ===" << std::endl;
    GGUFTokenizer tok;
    if (!tok.build_from(model)) {
        std::cerr << "  ❌ tokenizer 构建失败" << std::endl;
        return;
    }
    std::cout << "  词汇表大小: " << tok.size()
              << "  model=" << (tok.model_type.empty() ? "?" : tok.model_type)
              << "  bos=" << tok.bos_id << "  eos=" << tok.eos_id << std::endl;
    std::cout << "  前 5 个 token: ";
    for (std::int32_t i = 0; i < 5; ++i)
        std::cout << "[" << i << "]'" << tok.decode(i) << "' ";
    std::cout << std::endl;

    // 字节级 BPE 对任意 UTF-8 都无损（英文 / 中文往返）
    for (const char *text : {"Hello, world!", "你好，世界！"}) {
        const auto ids = tok.encode(text);
        std::string round;
        for (std::int32_t id : ids)
            round += tok.decode(id);
        std::cout << "  encode('" << text << "') → " << ids.size() << " tokens"
                  << "  decode(encode) = '" << round << "'"
                  << (round == text ? "  ✅ 往返一致" : "  （字节级，可能有空白差异）")
                  << std::endl;
    }
}

} // namespace

// ===========================================================================
// 入口：加载 → 按阶段依次演示
// ===========================================================================

int main(int argc, char **argv) {
    // 用法: ./main [model.gguf]（缺省用默认模型）
    const std::string model_path = argc > 1 ? argv[1] : kDefaultModelPath;

    GGUFModel model;
    if (!GGUFLoader::load(model_path, model)) {
        std::cerr << "❌ 加载 GGUF 文件失败: " << model_path << std::endl;
        return 1;
    }

    demo_header(model);                          // ① 文件头
    demo_metadata(model);                        // ② 元数据 KV
    demo_tensor_table(model);                    // ③ 张量信息表
    demo_data_region(model);                     // ③ 数据区
    demo_type_verification(model);               // 类型系统验证
    demo_direct_read(model, model_path);         // mmap 直接读取
    demo_dequantize(model, model_path);          // 反量化 + 交叉验证
    demo_weights_and_forward(model, model_path); // ④⑤⑥ 权重/前向/生成
    demo_tokenizer(model);                       // ③ 分词器
    return 0;
}
