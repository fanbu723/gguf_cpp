/*
 * GGMLSampler.hpp — 采样器（阶段 ⑥ 第 1 步）
 *
 * 从模型输出的 logits 中选下一个 token。支持：
 *  - GREEDY：直接取 argmax（贪心）
 *  - TOP_K：temperature 缩放 + softmax 后，只保留概率最大的前 k 个再采样
 *  - TOP_P：nucleus 采样，只保留累计概率 ≤ p 的最小集合再采样
 *  - TOP_K_P：top-k 与 top-p 组合（主流 LLM 默认，先 k 后 p）
 *
 * 采样有状态（内部随机数引擎推进），相同 seed 产生相同序列（可复现）。
 */

#pragma once

#include <cstdint>
#include <random>
#include <vector>

// 采样模式
enum class GGMLSampleMode {
    GREEDY = 0, // 贪心（temperature 无效）
    TOP_K,      // temperature + top-k
    TOP_P,      // temperature + top-p
    TOP_K_P,    // temperature + top-k + top-p（默认）
};

/**
 * @brief 采样器：logits → 下一个 token id
 */
struct GGMLSampler {
    GGMLSampleMode mode = GGMLSampleMode::TOP_K_P; // 采样模式
    float temperature = 0.8f;                      // 温度（>0；越小越确定）
    int top_k = 40;                                // top-k 保留数（≤0 表示禁用）
    float top_p = 0.9f;                            // top-p 累计概率阈值（<1 生效）
    std::uint64_t seed = 42;                       // 随机种子
    std::mt19937 rng;                              // 随机数引擎（随采样推进）

    /**
     * @brief 构造并设定种子
     */
    GGMLSampler() {
        reseed(seed);
    }

    /**
     * @brief 重置随机种子
     * @param s 新种子
     */
    void reseed(std::uint64_t s) {
        seed = s;
        rng.seed(static_cast<std::uint32_t>(s));
    }

    /**
     * @brief 从 logits 采样下一个 token id
     * @param logits 未归一化的 logits 向量
     * @return 选中的 token id
     */
    int sample(const std::vector<float> &logits);
};
