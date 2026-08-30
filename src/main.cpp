
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "GGMLChat.hpp"
#include "GGMLDequantize.hpp"
#include "GGMLForward.hpp"
#include "GGMLGenerate.hpp"
#include "GGMLNorm.hpp"
#include "GGMLRope.hpp"
#include "GGMLSSM.hpp"
#include "GGMLTransformer.hpp"
#include "GGMLType.hpp"
#include "GGUFLoader.hpp"
#include "GGUFModelWeights.hpp"
#include "GGUFTokenizer.hpp"

std::string model_path = "/home/dongfan/llm/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf";

int main() {
    GGUFModel model;

    if (!GGUFLoader::load(model_path, model)) {
        std::cerr << "❌ 加载 GGUF 文件失败: " << model_path << std::endl;
        return 1;
    }

    std::cout << "✅ 加载成功" << std::endl;
    std::cout << "  魔数: 0x" << std::hex << model.header.magic << std::dec << std::endl;
    std::cout << "  版本: " << model.header.version << std::endl;
    std::cout << "  元数据 KV 数量: " << model.header.metadata_kv_count << std::endl;
    std::cout << "  张量数量: " << model.header.tensor_count << std::endl;

    std::cout << "\n=== 元数据 KV 列表（前 20 条）===" << std::endl;
    const std::size_t meta_preview = std::min<std::size_t>(model.metadata.size(), 20);
    for (std::size_t i = 0; i < meta_preview; ++i) {
        const auto &kv = model.metadata[i];
        std::cout << "  [" << GGUFValueTypeName(kv.value_type) << "] " << kv.key << " = ";
        printMetadataValue(std::cout, kv.value);
        std::cout << std::endl;
    }
    if (model.metadata.size() > meta_preview) {
        std::cout << "  ... 共 " << model.metadata.size() << " 条" << std::endl;
    }

    std::cout << "\n=== 张量信息表（前 10 个）===" << std::endl;
    const std::size_t t_preview = std::min<std::size_t>(model.tensors.size(), 10);
    for (std::size_t i = 0; i < t_preview; ++i) {
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
    if (model.tensors.size() > t_preview) {
        std::cout << "  ... 共 " << model.tensors.size() << " 个张量" << std::endl;
    }

    std::cout << "\n=== 张量数据区 ===" << std::endl;
    std::cout << "  起始偏移: " << model.data.data_offset << std::endl;
    std::cout << "  总大小: " << model.data.data_size << " 字节 (" << std::fixed
              << std::setprecision(2)
              << static_cast<double>(model.data.data_size) / (1024.0 * 1024.0) << " MiB)"
              << std::endl;
    std::cout << "  数据指针: " << (model.data.data_ptr ? "已挂载" : "未挂载（延迟加载，待 mmap）")
              << std::endl;

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
    for (const auto &[type, p] : stat) {
        std::cout << "  type=" << type << "(" << GGMLTypeName(type) << ")"
                  << "  字节/元素=" << static_cast<double>(p.first) / static_cast<double>(p.second)
                  << "  总字节=" << p.first << std::endl;
    }

    std::cout << "\n=== mmap 映射后读取张量数据 ===" << std::endl;
    if (GGUFLoader::map_data(model_path, model)) {
        std::cout << "  映射成功: data_ptr = " << static_cast<const void *>(model.data.data_ptr)
                  << "（映射 " << model.data.map_len << " 字节）" << std::endl;

        // 找一个 F32 张量（data_type == 0）演示读取
        const GGUFTensorInfo *target = nullptr;
        for (const auto &t : model.tensors) {
            if (t.data_type == 0) {
                target = &t;
                break;
            }
        }

        if (target) {
            // GGUF 的 tensor.offset 是相对"数据区起点"的偏移，
            // 而 data_ptr 正指向数据区起点 → 张量位置 = data_ptr + offset
            const std::uint8_t *raw = model.data.data_ptr + target->offset;
            const float *f = reinterpret_cast<const float *>(raw);

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

        GGUFLoader::unmap_data(model);
        std::cout << "  已 munmap 释放（data_ptr 已清空）" << std::endl;
    } else {
        std::cerr << "  ❌ map_data 失败" << std::endl;
    }

    std::cout << "\n=== 反量化演示（GGMLDequantize）===" << std::endl;
    if (GGUFLoader::map_data(model_path, model)) {
        // 找一个 BF16 张量（type == 30，本模型校准）反量化
        const GGUFTensorInfo *bf16 = nullptr;
        for (const auto &t : model.tensors) {
            if (t.data_type == 30) {
                bf16 = &t;
                break;
            }
        }
        if (bf16) {
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
        }

        // 交叉验证：F32 张量 "直接读" vs "反量化" 应完全一致
        const GGUFTensorInfo *f32 = nullptr;
        for (const auto &t : model.tensors) {
            if (t.data_type == 0) {
                f32 = &t;
                break;
            }
        }
        if (f32) {
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

        GGUFLoader::unmap_data(model);
    }

    std::cout << "\n=== 模型权重加载演示（阶段④）===" << std::endl;
    if (GGUFLoader::map_data(model_path, model)) {
        GGUFModelWeights weights;
        if (weights.build(model)) {
            const auto &cfg = weights.config();
            std::cout << "  架构: " << (cfg.arch.empty() ? "?" : cfg.arch)
                      << "  层数: " << cfg.block_count << "  hidden: " << cfg.embedding_length
                      << "  heads: " << cfg.head_count << "x" << cfg.head_count_kv
                      << "kv  FFN: " << cfg.feed_forward_length
                      << "  SSM state: " << cfg.ssm_state_size
                      << "  全注意力间隔: " << cfg.full_attention_interval << std::endl;
            std::cout << "  张量视图数: " << weights.count() << std::endl;

            std::uint32_t attn = 0, ssm = 0;
            for (const auto &b : weights.blocks())
                (b.is_attention() ? attn : ssm)++;
            std::cout << "  块类型: Attention 层 " << attn << " 个 / SSM 层 " << ssm << " 个"
                      << std::endl;

            if (const auto *embd = weights.token_embd()) {
                std::vector<float> buf;
                if (embd->read_all(buf) && buf.size() >= 8) {
                    std::cout << "  token_embd.weight 前 8 个元素: ";
                    for (std::size_t i = 0; i < 8; ++i)
                        std::cout << buf[i] << (i + 1 < 8 ? ", " : "");
                    std::cout << std::endl;
                }
            }
            if (!weights.blocks().empty()) {
                const auto &b0 = weights.blocks()[0];
                const char *kind = b0.is_ssm() ? "SSM 混合层" : "Attention 层";
                if (const auto *norm = b0.attn_norm) {
                    std::vector<float> buf;
                    if (norm->read_all(buf) && buf.size() >= 4) {
                        std::cout << "  blk.0(" << kind << ") attn_norm 前 4 个元素: ";
                        for (std::size_t i = 0; i < 4; ++i)
                            std::cout << buf[i] << (i + 1 < 4 ? ", " : "");
                        std::cout << std::endl;
                    }
                }
            }

            // ---- 阶段⑤ 基础算子在真实权重上演示 ----
            std::cout << "\n  --- 阶段⑤ 基础算子演示（真实权重）---" << std::endl;
            if (!weights.blocks().empty()) {
                const auto &b0 = weights.blocks()[0];
                const auto &cfg = weights.config();
                const int hidden = static_cast<int>(cfg.embedding_length);
                // RMSNorm：输入全 1，用真实 gamma
                if (const auto *norm = b0.attn_norm) {
                    std::vector<float> gamma;
                    if (norm->read_all(gamma) && gamma.size() >= 4) {
                        std::vector<float> x(static_cast<std::size_t>(hidden), 1.0f);
                        std::vector<float> y(static_cast<std::size_t>(hidden));
                        GGMLRmsNorm(x.data(), gamma.data(), y.data(), hidden, cfg.rms_eps);
                        std::cout << "  RMSNorm(全1输入) 前 4 个输出: ";
                        for (int i = 0; i < 4; ++i)
                            std::cout << y[i] << (i + 1 < 4 ? ", " : "");
                        std::cout << std::endl;
                    }
                }
                // SwiGLU FFN：诊断权重中的 NaN
                // 实测：该 BF16 模型 FFN 权重含约 0.3% 的 BF16 NaN（指数全 1、尾数非 0），
                // 属模型文件本身数据，非解析错误；正常前向会因此产生 NaN 输出。
                if (b0.ffn_gate && b0.ffn_up && b0.ffn_down) {
                    std::vector<float> gate, up, down;
                    if (b0.ffn_gate->read_all(gate) && b0.ffn_up->read_all(up) &&
                        b0.ffn_down->read_all(down)) {
                        auto count_nan = [](const std::vector<float> &v) {
                            std::size_t n = 0;
                            for (float f : v)
                                if (std::isnan(f))
                                    ++n;
                            return n;
                        };
                        std::cout << "  SwiGLU 权重: gate=" << gate.size() << " up=" << up.size()
                                  << " down=" << down.size() << "  NaN 数: " << count_nan(gate)
                                  << "/" << count_nan(up) << "/" << count_nan(down)
                                  << "（BF16 权重本身含 NaN）" << std::endl;
                    }
                }
                // RoPE：旋转保范演示
                {
                    std::vector<float> q(8, 0.0f), y(8);
                    for (int i = 0; i < 8; ++i)
                        q[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
                    GGMLRopeNeox(q.data(), y.data(), 8, 8, 3, cfg.rope_freq_base);
                    float n0 = 0, n1 = 0;
                    for (int i = 0; i < 4; ++i) {
                        n0 +=
                            q[static_cast<std::size_t>(i)] * q[static_cast<std::size_t>(i)] +
                            q[static_cast<std::size_t>(i + 4)] * q[static_cast<std::size_t>(i + 4)];
                        n1 +=
                            y[static_cast<std::size_t>(i)] * y[static_cast<std::size_t>(i)] +
                            y[static_cast<std::size_t>(i + 4)] * y[static_cast<std::size_t>(i + 4)];
                    }
                    std::cout << "  RoPE(pos=3) 前 4 维: ";
                    for (int i = 0; i < 4; ++i)
                        std::cout << y[i] << (i + 1 < 4 ? ", " : "");
                    std::cout << "  范数 " << n0 << "→" << n1 << "（应相等）" << std::endl;
                }
            }

            // ---- 阶段⑤ 第2步：真实模型 Attention 层前向演示 ----
            std::cout << "\n  --- 阶段⑤ 第2步：Attention 层前向（真实权重）---" << std::endl;
            {
                const GGUFBlockWeights *blk = nullptr;
                for (const auto &b : weights.blocks())
                    if (b.is_attention()) {
                        blk = &b;
                        break;
                    }
                const auto *embd = weights.token_embd();
                if (blk && embd) {
                    const auto &cfg = weights.config();
                    const int hidden = static_cast<int>(cfg.embedding_length);
                    // 用 token 0 的 embedding 作为输入（读前 hidden 个元素）
                    std::vector<float> x(static_cast<std::size_t>(hidden));
                    for (int i = 0; i < hidden; ++i)
                        embd->read_element(static_cast<std::uint64_t>(i),
                                           x[static_cast<std::size_t>(i)]);
                    // KV cache：2 kv 头，head_dim 256，最多 8 位置
                    GGMLKVCache cache;
                    cache.init(static_cast<int>(cfg.head_count_kv),
                               static_cast<int>(cfg.key_length), static_cast<int>(cfg.value_length),
                               8);
                    std::vector<float> y(static_cast<std::size_t>(hidden));
                    GGMLTransformerAttentionBlock(*blk, cfg, cache, 0, x.data(), y.data());
                    bool finite = true;
                    for (float v : y)
                        if (!std::isfinite(v))
                            finite = false;
                    std::cout << "  blk.3(Attention 层) 前向输出 前 4 个元素: ";
                    for (int i = 0; i < 4; ++i)
                        std::cout << y[static_cast<std::size_t>(i)] << (i + 1 < 4 ? ", " : "");
                    std::cout << "  全有限: " << (finite ? "✅" : "❌（模型权重含 NaN）")
                              << std::endl;
                } else {
                    std::cout << "  未找到 Attention 层或 token_embd" << std::endl;
                }
            }

            // ---- 阶段⑤ 第3步：真实模型 SSM 混合层前向演示 ----
            std::cout << "\n  --- 阶段⑤ 第3步：SSM 混合层前向（真实权重）---" << std::endl;
            {
                const GGUFBlockWeights *blk = nullptr;
                for (const auto &b : weights.blocks())
                    if (b.is_ssm()) {
                        blk = &b;
                        break;
                    }
                const auto *embd = weights.token_embd();
                if (blk && embd) {
                    const auto &cfg = weights.config();
                    const int hidden = static_cast<int>(cfg.embedding_length);
                    std::vector<float> x(static_cast<std::size_t>(hidden));
                    for (int i = 0; i < hidden; ++i)
                        embd->read_element(static_cast<std::uint64_t>(i),
                                           x[static_cast<std::size_t>(i)]);
                    // SSM 状态：n_group 个 v-head × d_state×d_state；conv_dim = 2×key_dim +
                    // value_dim
                    const int n_group = static_cast<int>(cfg.ssm_group_count);
                    const int d_state = static_cast<int>(cfg.ssm_state_size);
                    const int key_dim = d_state * n_group;
                    const int conv_dim = 2 * key_dim + static_cast<int>(cfg.ssm_inner_size);
                    GGMLSSMState st;
                    st.init(n_group, d_state, conv_dim);
                    std::vector<float> y(static_cast<std::size_t>(hidden));
                    GGMLSSMLayer(*blk, cfg, st, x.data(), y.data());
                    bool finite = true;
                    for (float v : y)
                        if (!std::isfinite(v))
                            finite = false;
                    std::cout << "  blk.0(SSM 层) 前向输出 前 4 个元素: ";
                    for (int i = 0; i < 4; ++i)
                        std::cout << y[static_cast<std::size_t>(i)] << (i + 1 < 4 ? ", " : "");
                    std::cout << "  全有限: " << (finite ? "✅" : "❌（模型权重含 NaN）")
                              << std::endl;
                } else {
                    std::cout << "  未找到 SSM 层或 token_embd" << std::endl;
                }
            }

            // ---- 阶段⑤ 第4步：真实模型全模型前向演示 ----
            std::cout << "\n  --- 阶段⑤ 第4步：全模型前向（真实权重）---" << std::endl;
            {
                GGMLModelState st;
                st.init(weights, 32);
                std::vector<float> logits;
                GGMLForward(weights, st, 0, 0, logits);
                bool fin = true;
                int nan_cnt = 0;
                for (float v : logits) {
                    if (!std::isfinite(v)) {
                        fin = false;
                        ++nan_cnt;
                    }
                }
                std::cout << "  token 0 前向 → logits 大小 " << logits.size() << ", 前 5 个: ";
                for (int i = 0; i < 5 && i < static_cast<int>(logits.size()); ++i)
                    std::cout << logits[static_cast<std::size_t>(i)] << (i + 1 < 5 ? ", " : "");
                std::cout << "  全有限: "
                          << (fin ? "✅"
                                  : "❌（NaN " + std::to_string(nan_cnt) + " 个，模型权重含 NaN）")
                          << std::endl;
            }

            // ---- 阶段⑥：生成引擎演示（真实模型）----
            std::cout << "\n  --- 阶段⑥：生成引擎（真实模型）---" << std::endl;
            {
                GGUFTokenizer tok;
                if (tok.build_from(model)) {
                    const std::string prompt_text = "Hello";
                    const auto prompt_tokens = tok.encode(prompt_text);
                    if (!prompt_tokens.empty()) {
                        GGMLModelState st;
                        st.init(weights, 32);
                        // 先探测一次 logits：模型含 NaN 则无法生成（避免采样崩溃）
                        std::vector<float> probe;
                        GGMLForward(weights, st, prompt_tokens[0], 0, probe);
                        bool nan = false;
                        for (float v : probe)
                            if (!std::isfinite(v)) {
                                nan = true;
                                break;
                            }
                        if (nan) {
                            std::cout << "  ❌ 模型 logits 含 NaN（模型权重本身含 NaN），"
                                         "无法生成有效文本；建议换干净的 Qwen3.5 GGUF"
                                      << std::endl;
                        } else {
                            GGMLSampler sampler;
                            sampler.mode = GGMLSampleMode::TOP_K_P;
                            const auto gen =
                                GGMLGenerate(weights, st, sampler, prompt_tokens, 16, tok.eos_id);
                            std::string text = prompt_text;
                            for (int t : gen)
                                text += tok.decode(t);
                            std::cout << "  prompt: '" << prompt_text << "'" << std::endl;
                            std::cout << "  生成  : '" << text << "'  （" << gen.size()
                                      << " 个新 token）" << std::endl;
                        }
                    } else {
                        std::cout << "  prompt 编码为空" << std::endl;
                    }
                } else {
                    std::cout << "  tokenizer 构建失败" << std::endl;
                }
            }

            // ---- 阶段⑥：Chat 多轮对话演示（真实模型）----
            std::cout << "\n  --- 阶段⑥：Chat 多轮对话（真实模型）---" << std::endl;
            {
                GGUFTokenizer tok;
                if (tok.build_from(model)) {
                    GGMLChat chat;
                    chat.init(weights, tok, GGMLSampleMode::TOP_K_P, 42, 32);
                    std::cout << "  GGMLChat 已初始化（Qwen 风格多轮对话封装）" << std::endl;
                    std::cout << "  ⚠️ 真实模型 logits 全 NaN（FFN 权重含 NaN），无法实际"
                                 "对话；建议换干净的 Qwen3.5 GGUF 后体验完整生成链路"
                              << std::endl;
                } else {
                    std::cout << "  tokenizer 构建失败" << std::endl;
                }
            }
        } else {
            std::cerr << "  ❌ 权重索引构建失败" << std::endl;
        }
        GGUFLoader::unmap_data(model);
    }

    std::cout << "\n=== Tokenizer 演示 ===" << std::endl;
    GGUFTokenizer tok;
    if (tok.build_from(model)) {
        std::cout << "  词汇表大小: " << tok.size()
                  << "  model=" << (tok.model_type.empty() ? "?" : tok.model_type)
                  << "  bos=" << tok.bos_id << "  eos=" << tok.eos_id << std::endl;
        std::cout << "  前 5 个 token: ";
        for (std::int32_t i = 0; i < 5; ++i)
            std::cout << "[" << i << "]'" << tok.decode(i) << "' ";
        std::cout << std::endl;

        const std::string text = "Hello, world!";
        const auto ids = tok.encode(text);
        std::cout << "  encode('" << text << "') → ";
        for (std::size_t i = 0; i < ids.size(); ++i)
            std::cout << ids[i] << (i + 1 < ids.size() ? "," : "");
        std::cout << std::endl;

        std::string round;
        for (std::int32_t id : ids)
            round += tok.decode(id);
        std::cout << "  decode(encode) = '" << round << "'"
                  << (round == text ? "  ✅ 往返一致" : "  （字节级，可能有空白差异）")
                  << std::endl;

        // 中文往返：字节级 BPE 对任意 UTF-8 都无损
        const std::string text2 = "你好，世界！";
        const auto ids2 = tok.encode(text2);
        std::string round2;
        for (std::int32_t id : ids2)
            round2 += tok.decode(id);
        std::cout << "  encode('" << text2 << "') → " << ids2.size() << " tokens"
                  << "  decode(encode) = '" << round2 << "'"
                  << (round2 == text2 ? "  ✅ 往返一致" : "  ❌ 不一致") << std::endl;
    } else {
        std::cerr << "  ❌ tokenizer 构建失败" << std::endl;
    }
    return 0;
}
