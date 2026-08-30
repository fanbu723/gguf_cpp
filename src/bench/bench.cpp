/*
 * bench.cpp — 性能测试（统计运行消耗与性能情况）
 *
 * 对真实模型统计各阶段的耗时与吞吐：
 *  - 解析 / mmap / Tokenizer / 权重索引 的耗时
 *  - 单层前向（SSM / Attention，含权重反量化）耗时
 *  - 全模型前向：平均耗时、ms/token、tokens/s
 *  - 内存占用（VmRSS）
 *
 * 用法：./build/bench [forward_runs]
 *   forward_runs：全模型前向运行次数（默认 3，取平均）
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "GGMLForward.hpp"
#include "GGMLSSM.hpp"
#include "GGMLTransformer.hpp"
#include "GGUFLoader.hpp"
#include "GGUFModelWeights.hpp"
#include "GGUFTokenizer.hpp"

namespace {
std::string model_path = "/home/dongfan/llm/Qwen3.5-0.8B-clean-BF16.gguf";

using Clock = std::chrono::steady_clock;
double now_ms() {
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

long rss_kb() { // 当前物理内存占用（KB），读 /proc/self/status
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            long kb = 0;
            std::sscanf(line.c_str(), "VmRSS: %ld kB", &kb);
            return kb;
        }
    }
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    int fwd_runs = 3;
    if (argc > 1)
        fwd_runs = std::atoi(argv[1]);
    if (fwd_runs < 1)
        fwd_runs = 1;

    std::cout << "=== GGUF 推理性能测试 ===" << std::endl;
    std::cout << "  模型: " << model_path << std::endl;
    std::cout << "  全模型前向次数: " << fwd_runs << std::endl << std::endl;

    double t;
    GGUFModel model;

    // 1. 解析
    t = now_ms();
    if (!GGUFLoader::load(model_path, model)) {
        std::cerr << "❌ 加载 GGUF 失败" << std::endl;
        return 1;
    }
    const double t_load = now_ms() - t;

    // 2. mmap
    t = now_ms();
    GGUFLoader::map_data(model_path, model);
    const double t_map = now_ms() - t;

    // 3. Tokenizer
    GGUFTokenizer tok;
    t = now_ms();
    tok.build_from(model);
    const double t_tok = now_ms() - t;

    // 4. 权重索引
    GGUFModelWeights weights;
    t = now_ms();
    weights.build(model);
    const double t_w = now_ms() - t;

    const auto &cfg = weights.config();
    const int hidden = static_cast<int>(cfg.embedding_length);
    std::cout << "  模型: " << (cfg.arch.empty() ? "?" : cfg.arch) << "  " << cfg.block_count
              << " 层 (" << cfg.embedding_length << " hidden, vocab "
              << weights.token_embd()->dims[1] << ")" << std::endl
              << std::endl;

    // 5. 单层前向（各 1 次，含权重反量化）
    const GGUFBlockWeights *ssm_blk = nullptr, *attn_blk = nullptr;
    for (const auto &b : weights.blocks()) {
        if (!ssm_blk && b.is_ssm())
            ssm_blk = &b;
        if (!attn_blk && b.is_attention())
            attn_blk = &b;
    }
    std::vector<float> x(static_cast<std::size_t>(hidden), 0.1f);
    std::vector<float> y(static_cast<std::size_t>(hidden));

    double t_ssm = 0, t_attn = 0;
    if (ssm_blk) {
        const int n_group = static_cast<int>(cfg.ssm_group_count);
        const int d_state = static_cast<int>(cfg.ssm_state_size);
        const int key_dim = d_state * n_group;
        const int conv_dim = 2 * key_dim + static_cast<int>(cfg.ssm_inner_size);
        GGMLSSMState st;
        st.init(n_group, d_state, conv_dim);
        t = now_ms();
        GGMLSSMLayer(*ssm_blk, cfg, st, x.data(), y.data());
        t_ssm = now_ms() - t;
    }
    if (attn_blk) {
        GGMLKVCache kv;
        kv.init(static_cast<int>(cfg.head_count_kv), static_cast<int>(cfg.key_length),
                static_cast<int>(cfg.value_length), 8);
        t = now_ms();
        GGMLTransformerAttentionBlock(*attn_blk, cfg, kv, 0, x.data(), y.data());
        t_attn = now_ms() - t;
    }

    // 6. 全模型前向（多次取平均）
    double t_fwd_sum = 0;
    for (int i = 0; i < fwd_runs; ++i) {
        GGMLModelState st;
        st.init(weights, 32);
        std::vector<float> logits;
        t = now_ms();
        GGMLForward(weights, st, 0, 0, logits);
        t_fwd_sum += now_ms() - t;
        if (i == 0 && !logits.empty()) {
            bool nan = false;
            for (float v : logits)
                if (!std::isfinite(v)) {
                    nan = true;
                    break;
                }
            if (nan)
                std::cout << "  ⚠️ 注意：logits 含 NaN（模型权重本身含 NaN），"
                             "生成链路不可用；以下为纯计算性能数据"
                          << std::endl;
        }
    }
    const double t_fwd_avg = t_fwd_sum / fwd_runs;

    const long rss = rss_kb();

    // ---- 报告 ----
    std::cout << "\n===== 各阶段耗时 =====" << std::endl;
    auto fmt = [](const char *name, double ms) { std::printf("  %-28s %10.1f ms\n", name, ms); };
    fmt("解析 (GGUFLoader::load)", t_load);
    fmt("mmap 映射 (map_data)", t_map);
    fmt("Tokenizer 构建", t_tok);
    fmt("权重索引 (GGUFModelWeights)", t_w);
    fmt("单层前向 SSM (含反量化)", t_ssm);
    fmt("单层前向 Attention (含反量化)", t_attn);

    std::printf("\n===== 全模型前向（%d 次平均）=====\n", fwd_runs);
    std::printf("  每次耗时      : %10.1f ms\n", t_fwd_avg);
    std::printf("  单 token 耗时 : %10.1f ms/token\n", t_fwd_avg);
    std::printf("  吞吐          : %10.2f tokens/s\n", 1000.0 / t_fwd_avg);

    std::printf("\n===== 内存占用 =====");
    std::printf("\n  VmRSS (常驻物理内存): %ld KB (%.2f MB)\n", rss,
                static_cast<double>(rss) / 1024.0);

    std::printf("\n===== 性能画像说明 =====");
    std::printf("\n  瓶颈：全模型前向每次重新反量化所有权重（BF16→float），"
                "且 logits 逐元素反量化 embedding（vocab×hidden 次）");
    std::printf("\n  优化方向：权重缓存、logits 用批量反量化 / SIMD、预填充并行化\n");
    return 0;
}
